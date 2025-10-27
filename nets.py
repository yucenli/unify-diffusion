from omegaconf import OmegaConf

from src.base_models.unet import KingmaUNet
from src.base_models.protein_convnet import ByteNetLMTimeNew
from src.base_models.faesm import FAESM_Base
from src.base_models.dit_text import DIT
import transformers
from src.base_models.dna_model import CNNModel as DNAConv

image_nn_name_dict = {
    "KingmaUNet": KingmaUNet,
}

language_nn_name_dict = {
    "DIT": DIT,
}

protein_nn_name_dict = {
    "ConvNew": ByteNetLMTimeNew,
    "FAESM": FAESM_Base,
}

dna_nn_name_dict = {
    "DNAConv": DNAConv,
}

def get_model_setup(cfg, tokenizer=None):

    schedule_conditioning = cfg.model.model in [
        "ScheduleCondition", "DiscreteScheduleCondition",
        "MaskingDiffusion",
    ]

    nn_params = cfg.architecture.nn_params
    nn_params = (OmegaConf.to_container(nn_params, resolve=True)
            if nn_params is not None else {})

    if cfg.architecture.x0_model_class in image_nn_name_dict:

        nn_params = {
            "n_channel": 1 if cfg.data.data == 'MNIST' else 3, 
            "N": cfg.data.N + (cfg.model.model == 'MaskingDiffusion'),
            "n_T": cfg.model.n_T,
            "schedule_conditioning": schedule_conditioning,
            "s_dim": cfg.architecture.s_dim,
            **nn_params
        }

        return image_nn_name_dict[cfg.architecture.x0_model_class], nn_params
    
    elif cfg.architecture.x0_model_class in language_nn_name_dict:

        vocab_size = tokenizer.vocab_size
        if type(tokenizer) == transformers.models.byt5.tokenization_byt5.ByT5Tokenizer:
            vocab_size = 384


        nn_params = {
            "vocab_size": vocab_size,
            "schedule_conditioning": cfg.architecture.schedule_conditioning,
            "config": nn_params
        }
        return language_nn_name_dict[cfg.architecture.x0_model_class], nn_params
        
    elif cfg.architecture.x0_model_class in protein_nn_name_dict:
        nn_params = {
            "n_tokens": cfg.data.N + (cfg.model.model == 'MaskingDiffusion'),
            "schedule_conditioning": schedule_conditioning,
            **nn_params
        }
        return protein_nn_name_dict[cfg.architecture.x0_model_class], nn_params
    
    elif cfg.architecture.x0_model_class in dna_nn_name_dict:
        return dna_nn_name_dict[cfg.architecture.x0_model_class], cfg.architecture.nn_params
    
    else:
        raise NotImplementedError(f"Model {cfg.architecture.x0_model_class} not implemented")