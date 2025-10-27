import math
import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.models.distilbert.modeling_distilbert import Embeddings

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    try:
        import flash_attn.layers.rotary
        cos_new = cos[0,:,0,0,:cos.shape[-1]//2]
        sin_new = sin[0,:,0,0,:sin.shape[-1]//2]
        return flash_attn.layers.rotary.apply_rotary_emb_qkv_(
            qkv, cos_new, sin_new
        )
    except:
        # cos = cos.repeat(*([1] * (cos.dim()-1)), 2)[None,:,None,None,:]
        # sin = sin.repeat(*([1] * (sin.dim()-1)), 2)[None,:,None,None,:]
        return (qkv * cos) + (rotate_half(qkv) * sin)


# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=1000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    two_pi = 2 * 3.14159
    log_max_period = math.log(max_period)
    vec = torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) - half/3

    freqs = two_pi * torch.exp(-log_max_period * vec / half)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    # self.adaLN_modulation.weight.data.zero_()
    # self.adaLN_modulation.bias.data.zero_()


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, rotary_cos_sin, c, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    #WIP, allow for passing the schedule in as conditioning at each layer
    if len(c.shape) > 2:
      bs, seq_len = c.shape[0], c.shape[1]
      modulation = self.adaLN_modulation(
        c.reshape(-1, c.shape[-1])).reshape(bs, seq_len, -1)
    else:
      modulation = self.adaLN_modulation(c)[:, None]
    (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = modulation.chunk(6, dim=2)

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv,
                    'b s (three h d) -> b s three h d',
                    three=3,
                    h=self.n_heads)
    
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        
    xq, xk, xv = qkv.chunk(3, dim=2)
    
    x = F.scaled_dot_product_attention(
        xq.squeeze(2).permute(0, 2, 1, 3),
        xk.squeeze(2).permute(0, 2, 1, 3),
        xv.squeeze(2).permute(0, 2, 1, 3),
        dropout_p=0.0,
        is_causal=False,
    ).permute(0, 2, 1, 3)
    
    x = rearrange(x, 'b s h d -> b s (h d)', b=batch_size)

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x



class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    # self.adaLN_modulation.weight.data.zero_()
    # self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    if len(c.shape) > 2:
      bs, seq_len = c.shape[0], c.shape[1]
      modulation = self.adaLN_modulation(
        c.reshape(-1, c.shape[-1])).reshape(bs, seq_len, -1)
    else:
      modulation = self.adaLN_modulation(c)[:, None]
    
    shift, scale = modulation.chunk(2, dim=2)
    
    x = modulate_fused(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int, schedule_conditioning: bool):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    
    if schedule_conditioning:
        self.s_embed_input = TimestepEmbedder(config.hidden_size)
        self.s_embed_block = TimestepEmbedder(config.cond_dim)

    self.vocab_embed = EmbeddingLayer(config.hidden_size, vocab_size)
    self.sigma_map = TimestepEmbedder(config.cond_dim)
    self.rotary_emb = Rotary(config.hidden_size // config.n_heads)

    blocks = []
    for i in range(config.n_blocks):
      blocks.append(DDiTBlock(config.hidden_size,
                              config.n_heads,
                              config.cond_dim,
                              dropout=config.dropout))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.hidden_size,
      vocab_size,
      config.cond_dim)
    self.scale_by_sigma = config.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, indices, sigma, cond=None, S=None):
    """
    indices: x_t - perturbed input tokens - shape: (BSZ, max_len)

    indices: x_t - (B, L, V)
    sigma: diffusion time_step [0, t_max] - shape: (BSZ)
    cond: attention_mask - shape: (BSZ, max_len)
    S: number of substitution transformations
    """
    V = indices.shape[2]

    vocab = torch.arange(V).to(indices.device) #
    embeddings = self.vocab_embed(vocab)  
    embeddings_output = indices @ embeddings
    x = embeddings_output * cond.unsqueeze(-1)  # [B, L]

    c = F.silu(self.sigma_map(sigma))

    rotary_cos_sin = self.rotary_emb(x)

    if S is not None:
      bs, seq_len = S.shape[0], S.shape[1]
      S_out = F.silu(self.s_embed_input(S.reshape(-1)+1)).reshape(bs, seq_len, -1)
      x = x + S_out
      
      # WIP, this is approximately correct but not thoroughly tested
      S_out = F.silu(self.s_embed_block(S.reshape(-1)+1)).reshape(bs, seq_len, -1)
      c = c[:, None, :] + S_out
    
    try:
      with torch.amp.autocast(dtype=torch.bfloat16): #bfloat16
        for i in range(len(self.blocks)):
          x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
        x = self.output_layer(x, c)
    except:
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      x = self.output_layer(x, c)

    return x










