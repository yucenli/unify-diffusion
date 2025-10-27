from .esm import FAEsmForMaskedLM
from torch import nn

class FAESM_Base(nn.Module):
    def __init__(self, hf_model_name="esm2_t6_8M_UR50D", schedule_conditioning=True, **kwargs):
        super().__init__()
        print(f"Using FAESM model {hf_model_name}")
        conditioning_dim = kwargs.get("d_embedding", 128)
        pretrained = kwargs.get("pretrained", True)
        print('KWARGS use fa', kwargs.get("use_fa"))
        self.faesm = FAEsmForMaskedLM.from_pretrained(
            pretrained_model_name_or_path = f"facebook/{hf_model_name}",
            use_fa = kwargs.get("use_fa", True),
            conditioning_dim = conditioning_dim,
            load_pretrained_weights = pretrained
        )
        self.embed_dim = self.faesm.esm.embeddings.word_embeddings.embedding_dim # 320 for smallest ESM, 480 for 35M
        self.proj = nn.Linear(self.embed_dim, kwargs.get("n_classes")) 

    def forward(self, x, t, input_mask=None, S=None):
        cond = t if S is None else S
        embeddings = self.faesm(input_ids = x, attention_mask = input_mask, conditioning=cond)['last_hidden_state']
        preds = self.proj(embeddings).squeeze()
        return preds