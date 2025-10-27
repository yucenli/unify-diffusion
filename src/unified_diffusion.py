import torch.nn as nn
import random

from .trainer import ContinuousTimeDiffusion
import src


class UnifiedDiffusion(ContinuousTimeDiffusion):
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes: int = 10,
        schedule_type="cos",
        logistic_pars=False,
        model_list=["GaussianDiffusion", "SimplicialDiffusion", "DiscreteDiffusion"],
        ssp=True,
        **kwargs  # Pass all other arguments to parent class
    ) -> None:
        super().__init__(
            x0_model_class=x0_model_class,
            nn_params=nn_params,
            num_classes=num_classes,
            schedule_type=schedule_type,
            logistic_pars=logistic_pars,
            **kwargs
        )
        self.model_list = nn.ModuleList()
        for model_name in model_list:
            # initialize model 
            model_class = getattr(src, model_name) 
            model_instance = model_class(
                x0_model_class=x0_model_class,
                nn_params=nn_params,
                num_classes=num_classes,
                schedule_type=schedule_type,
                logistic_pars=logistic_pars,
                ssp=True,
                **kwargs
            )
            model_instance.x0_model = self.x0_model
            self.model_list.append(model_instance)

    def pre_configure_model(self, train_dataloader, val_dataloader=None):
        for model in self.model_list:
            model.pre_configure_model(train_dataloader, val_dataloader)
    
    def training_step(self, batch, batch_idx):
        x, attn_mask, cache = self.get_inputs(batch)

        # choose between Gaussian, discrete, simplical 
        model = random.choice(self.model_list)
        loss, info = model(x, attn_mask, extra=cache)
        print(f"{model.__class__.__name__} loss {loss}")

        if self.sample_x is None:
            self.sample_x = x[:1]
            self.sample_a = None if attn_mask is None else attn_mask[:1]
                
        self.log('train_loss', info['vb_loss'], sync_dist=True)
        self.log('train_ce_loss', info['ce_loss'], sync_dist=True)
        self.log(f'train_loss_{model.__class__.__name__}', info['vb_loss'], sync_dist=True)
        self.log(f'train_ce_loss_{model.__class__.__name__}', info['ce_loss'], sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):        
        x, attn_mask, cache = self.get_inputs(batch)

        # get loss for each model 
        avg_vb_loss = 0
        avg_ce_loss = 0
        
        loss_dict = {}
        for model in self.model_list:
            loss, info = model(x, attn_mask, extra=cache)
            loss_dict[f'val_loss_{model.__class__.__name__}'] = info['vb_loss']
            loss_dict[f'val_ce_loss_{model.__class__.__name__}'] = info['ce_loss']
            avg_vb_loss += info['vb_loss']
            avg_ce_loss += info['ce_loss']
        avg_vb_loss = avg_vb_loss/len(self.model_list)
        avg_ce_loss = avg_ce_loss/len(self.model_list)
        
        loss_dict['val_loss'] = avg_vb_loss
        loss_dict['val_ce_loss'] = avg_ce_loss
        self.log_dict(loss_dict, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss_dict
    
    def sample(self, B, D, delta_t, device='cuda', max_D=None, **kwargs):
        samples = {}
        for model in self.model_list:
            model.eval()
            model_samples = model.sample(B, D, delta_t, device=device, max_D=max_D, **kwargs)
            samples[model.__class__.__name__] = model_samples            
        return samples