# Code adapted from https://github.com/AlanNawzadAmin/SCUD/blob/main/scud/trainer.py

import tempfile

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.amp import autocast
import pytorch_lightning as pl
from torchvision.utils import make_grid
from transformers import get_constant_schedule_with_warmup

from .utils.schedule_sample import sample_n_transitions_cont


def get_gif(sample_x, sample_a, model, gen_trans_step, batch_size):
    # save images
    p = model.get_stationary()
    samples = torch.multinomial(p, num_samples=batch_size*sample_x.shape[1:].numel(), replacement=True)
    init_noise = samples.reshape((batch_size,)+sample_x.shape[1:]).to(sample_x.device)
    if sample_a is not None:
        attn_mask = sample_a.repeat(batch_size, *[1]*(sample_a.dim()-1))
    else:
        attn_mask = None
    images = model.sample_sequence(
        init_noise, attn_mask, stride=3, n_T=gen_trans_step,
    )
    if images is not None:
        # image sequences to gif
        gif = []
        for image in images:
            x_as_image = make_grid(image.float() / (model.num_classes - 1), nrow=2)
            img = x_as_image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            gif.append(Image.fromarray(img))

        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
            gif[0].save(
                temp_file.name,
                format='GIF',
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file_img:
            last_img = gif[-1]
            last_img.save(temp_file_img)
        return temp_file.name, temp_file_img.name
    else: 
        return None, None

def get_text(sample_x, sample_a, model, gen_trans_step, batch_size, tokenizer):
    # save images
    p = model.get_stationary()
    samples = torch.multinomial(p, num_samples=batch_size*sample_x.shape[1:].numel(), replacement=True)
    init_noise = samples.reshape((batch_size,)+sample_x.shape[1:]).to(sample_x.device)
    if sample_a is not None:
        attn_mask = sample_a.repeat(batch_size, *[1]*(sample_a.dim()-1))
    else:
        attn_mask = None
    tokens = model.sample_sequence(
        init_noise, attn_mask, stride=3, n_T=gen_trans_step,
    )
    if tokens is not None:
        last_token = tokens[-1]
        stride_tokens = tokens[::(gen_trans_step // 3)//10 + 1]
        if sample_a is not None:
            if hasattr(tokenizer, 'pad_id'):
                pad_id = tokenizer.pad_id
            elif hasattr(tokenizer, 'pad_token_id'):
                pad_id = tokenizer.pad_token_id
            last_token[attn_mask == 0.] = pad_id
            for t in stride_tokens:
                t[attn_mask == 0.] = pad_id
        if hasattr(tokenizer, 'decode'):
            dt = lambda tok: [tokenizer.decode(t) for t in tok]
        elif hasattr(tokenizer, 'untokenize'):
            dt = lambda tok: [tokenizer.untokenize(t)[:int(a.sum())]
                              for t, a in zip(tok,attn_mask)]
        return dt(last_token), [dt(t) for t in stride_tokens]
    else:
        return None


class DiffusionTrainer(pl.LightningModule):
    def __init__(self, lr=1e-3, gen_trans_step=1000, n_gen_images=4, grad_clip_val=1, weight_decay=0, seed=0, n_stat_samples=4e4, tokenizer=None, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.weight_decay = weight_decay
        # logging
        self.sample_x = None
        self.validation_step_outputs = []
        self.gen_trans_step = gen_trans_step
        self.n_gen_images = n_gen_images
        self.n_stat_samples = n_stat_samples
        self.tokenizer = tokenizer
        # self.to(torch.float32)

    def forward(self, x):
        return NotImplementedError

    # def get_kl_t1(self, x):
    #     return NotImplementedError

    def pre_configure_model(self, dataloader, test_dataloader=None):
        pass

    def calc_p0(self, dataloader):
        # get stationary dist
        p0 = torch.ones(self.num_classes)
        pbar = tqdm(total=self.n_stat_samples)
        for i, batch in tqdm(enumerate(dataloader)):  # (16, 1024)
            if p0.sum() > self.n_stat_samples:  
                break
            x, _, _ = self.get_inputs(batch)
            
            new = F.one_hot(x.long(), num_classes=self.num_classes).to(torch.float32).view((-1, self.num_classes)).sum(0)
            p0 = p0 + new
            pbar.update(new.sum().item())

        pbar.close()
        p0 = p0 / p0.sum()
        self.register_buffer("p0", p0)
        
    def get_inputs(self, batch):
        wf_cache = None
        if isinstance(batch, tuple) or isinstance(batch, list):
            if len(batch) == 2: #protein datasets
                x, attn_mask = batch 
            elif len(batch) == 3: # cached datasets
                x, attn_mask, wf_cache = batch
            else:
                raise ValueError("Batch parsing error.")
        elif isinstance(batch, dict): #text datasets
            x, attn_mask = batch['input_ids'], batch['attention_mask']
        else: #image datasets
            x = batch
            attn_mask = None
        return x, attn_mask, wf_cache

    def training_step(self, batch, batch_idx):
        x, attn_mask, wf_cache = self.get_inputs(batch)
        loss, info = self(x, attn_mask, wf_cache)
        if self.sample_x is None:
            self.sample_x = x[:1]
            self.sample_a = None if attn_mask is None else attn_mask[:1]
        
        self.log_dict({
            "train_ce_loss": info['ce_loss'],
            "train_loss": info['vb_loss'],
            "train_perplexity": np.exp(info['ce_loss']),
        })

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_inputs(batch)

        loss, info = self(*inputs)
        loss_dict = {
            "val_ce_loss": info['ce_loss'],
            "val_loss": info['vb_loss'],
            "val_perplexity": np.exp(info['ce_loss']),
        }
        self.log_dict(loss_dict, on_step=False, on_epoch=True, sync_dist=True)
        return loss_dict

    # @rank_zero_only
    # def on_validation_epoch_end(self,):
    #     # generate image
    #     if self.sample_x is not None:
    #         with torch.no_grad():
    #             if self.tokenizer is None:
    #                 gif_fname, img_fname = get_gif(self.sample_x, self.sample_a, self, self.gen_trans_step, self.n_gen_images)
    #                 if gif_fname is not None:
    #                     if isinstance(self.logger, pl.loggers.WandbLogger):
    #                         wandb.log({"sample_gif": wandb.Image(gif_fname)})
    #                         wandb.log({"sample_gif_last": wandb.Image(img_fname)})
    #             else:
    #                 print("getting text")
    #                 last_text, gen_text = get_text(self.sample_x, self.sample_a, self, self.gen_trans_step, self.n_gen_images, self.tokenizer)
    #                 if last_text is not None:
    #                     if isinstance(self.logger, pl.loggers.WandbLogger):
    #                         joined_text = "\n\n".join(last_text)
    #                         wandb.log({"sample_text": wandb.Table(columns=["text"], data=[[joined_text]])})
    #                         joined_text_gen = ["\n\n".join(t) for t in gen_text]
    #                         wandb.log({"sample_text_process": wandb.Table(columns=["text"], data=[[jt] for jt in joined_text_gen])})

    # def on_fit_start(self):
    #     if isinstance(self.logger, pl.loggers.WandbLogger):
    #         wandb.config.update(self.hparams)

    def on_before_optimizer_step(self, optimizer):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_val)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=2500
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


def get_betas(schedule_type): # TODO remove probably - artifact from SCUD paper
    if schedule_type in ['cos', 'linear']:
        def get_funcs(L, p0, model='SEDD', scale=1, type_=None):
            if schedule_type == 'cos':
                alpha = lambda t: 1-torch.cos((1 - t) * torch.pi / 2)
                alpha_prime = lambda t: -torch.sin((1 - t) * torch.pi / 2) * torch.pi / 2
            if schedule_type == 'linear':
                alpha = lambda t: 1-t
                alpha_prime = lambda t: -1
            beta = lambda t: - scale * alpha_prime(t) / alpha(t)
            log_alpha = lambda t: scale * torch.log(alpha(t))
            return log_alpha, beta
        return get_funcs
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    

class ContinuousTimeDiffusion(DiffusionTrainer):
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes: int = 10,
        schedule_type="cos",
        logistic_pars=False,
        t_max=0.999,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters(ignore=['x0_model_class'])
        self.hparams.update(x0_model_class=x0_model_class.__name__)
        self.x0_model = x0_model_class(**nn_params)
        self.eps = 1e-9
        self.num_classes = num_classes
        self.t_max = t_max
        self.logistic_pars = logistic_pars

        # Precalculate betas
        self.get_beta_func = get_betas(schedule_type)

    def base_predict(self, x_t, t, attn_mask, S=None):
        with autocast('cuda', dtype=torch.bfloat16):
            preds = self.x0_model(x_t, t, attn_mask, S)
        return preds.to(torch.float32)

    def model_predict(self, x_t, t, attn_mask, S=None):
        pred = self.base_predict(x_t, t, attn_mask, S)
        if not self.logistic_pars:
            return pred
        else:
            loc = pred[..., 0].unsqueeze(-1)
            log_scale = pred[..., 1].unsqueeze(-1)
            inv_scale = torch.exp(- (log_scale - 2.))
            bin_width = 2. / (self.num_classes - 1.)
            bin_centers = torch.linspace(-1., 1., self.num_classes).to(pred.device)
            bin_centers = bin_centers - loc
            log_cdf_min = torch.nn.LogSigmoid()(
                inv_scale * (bin_centers - 0.5 * bin_width))
            log_cdf_max = torch.nn.LogSigmoid()(
                inv_scale * (bin_centers + 0.5 * bin_width))
            logits = log_cdf_max + torch.log1p(-torch.exp(log_cdf_min-log_cdf_max)+self.eps)
            return logits

    def q_posterior_logits(self, x_0, x_t, t, S=None):
        raise NotImplementedError

    def x_t_sample(self, x_0, t, noise, S=None):
        raise NotImplementedError

    def sample_point(self, x, attn_mask=None, rand_shape=None):   
        # x is shape 
        t = torch.rand(x.shape[0], device=x.device) * self.t_max

        
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        x_t = self.x_t_sample(
            x, t, torch.rand((*x.shape, rand_shape if rand_shape is not None else self.num_classes), device=x.device), S
        )
        # if attn_mask is not None:
        #     x_t = torch.where(attn_mask==1, x_t, x)
        #     S = torch.where(attn_mask==1, S, 0 * S)
        return t, S, x_t

    def load_state_dict(self, state_dict, strict=False):
        # Call the parent class's load_state_dict method
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)

        # Load the additional state dict variables
        for key in ['p0_inds', 'p0_rank', 'K', 'L', 'K_coo', 'K_csc', 'K_T', 'L_T', 'stat', 'stationary']:
            if key in state_dict:
                setattr(self, key, state_dict[key])
                if key in unexpected_keys:
                    unexpected_keys.remove(key)
            # elif strict:
            #     missing_keys.append(key)

        if strict:
            error_msgs = []
            if len(unexpected_keys) > 0:
                error_msgs.append('unexpected key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.append('missing key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in missing_keys)))

            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                    self.__class__.__name__, "\n\t".join(error_msgs)))

        return missing_keys, unexpected_keys

    @classmethod
    def load_from_checkpoint_old(cls, checkpoint_path, map_location=None, **kwargs):
        print("Loading checkpoint ...")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        hparams = checkpoint['hyper_parameters']
        
        # Get the x0_model_class
        from src.base_models.unet import UNet, KingmaUNet, SimpleUNet, GigaUNet
        from src.dit_vision import DiT_Llama
        from src.base_models.dit_text import DIT
        from src.base_models.protein_convnet import ByteNetLMTime
        x0_model_class = {
            "SimpleUNet":SimpleUNet,
            "KingmaUNet":KingmaUNet,
            "UNet":UNet,
            "GigaUNet":GigaUNet,
            "DiT_Llama":DiT_Llama,
            "DIT": DIT,
            "ByteNetLMTime": ByteNetLMTime
        }[hparams['x0_model_class']]
        hparams['x0_model_class'] = x0_model_class

        # Create model
        print("Setting up class ...")
        model = cls(**hparams)
        print("Loading params ...")
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        print("Loading checkpoint ...")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        hparams = checkpoint['hyper_parameters']
        # Get the x0_model_class
        from src.base_models.faesm import FAESM_Base
        hparams['x0_model_class'] = FAESM_Base

        # Create model
        print("Setting up class ...")
        model = cls(**hparams)
        print("Loading params ...")
        model.load_state_dict(checkpoint['state_dict'])
        return model

