import math
import matplotlib.pyplot as plt
import os
import torch
import wandb

from .trainer import ContinuousTimeDiffusion
from .utils.utils import convert_to_probs


class GaussianDiffusion(ContinuousTimeDiffusion):
    """
    Gaussian diffusion

    Algorithm 2
    """
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes,
        forward_kwargs={"type":"uniform"},
        schedule_type="cos",
        logistic_pars=False,
        ssp=False,
        **kwargs
    ): 
        """
        Assumes L = (1 pi^T) - I
        """
        # Precalculate betas, define model_predict, p_sample
        super().__init__(
            x0_model_class=x0_model_class,
            nn_params=nn_params,
            num_classes=num_classes,
            forward_kwargs=forward_kwargs,
            schedule_type=schedule_type,
            logistic_pars=logistic_pars,
            dummy_L=True,
            **kwargs
        )            
        self.save_hyperparameters(ignore=['x0_model_class'])
        self.eps=1e-6
        self.num_classes = num_classes

        pi = self.get_stationary()
        self.register_buffer("pi", pi)

        self.tau, self.beta = self.calculate_tau()

        self.ssp = ssp
        self.rate = kwargs['rate']
    

    def calculate_tau(self):
        tau = lambda t: -1 * torch.log(1 - t) * self.rate
        tau_prime = lambda t:  1 / (1 - t) * self.rate     
        return tau, tau_prime
    

    def t_vs_loss_plot(self, x_sample, attn_mask, save_path, name):
        t, x_t = self.sample_point(x_sample)
        p_x_t_given_x_0 = self.p_x_t_given_x_0(x_t, t)
        pred_x0_logits = self.model_predict(x_t, t, attn_mask) + p_x_t_given_x_0
        loss, coeff = self.compute_loss_advanced(x_sample, pred_x0_logits, t)  # get B losses
        loss_t_coeff = (loss * coeff).mean(dim=-1)
        loss = loss.mean(dim=-1)
        ts = t.cpu().numpy()

        print(f"ts: {ts}")
        print(f"coeff: {coeff}")
        print(f"ts shape: {ts.shape}")
        print(f"coeff shape: {coeff.shape}")

        p_x_t_is_nan = torch.isnan(p_x_t_given_x_0).any()
        loss_is_nan = torch.isnan(loss).any()
        coeff_is_nan = torch.isnan(coeff).any()
        loss_t_coeff_is_nan = torch.isnan(loss_t_coeff).any()

        print(f"p_x_t_is_nan: {p_x_t_is_nan}")
        print(f"loss_is_nan: {loss_is_nan}")
        print(f"coeff_is_nan: {coeff_is_nan}")
        print(f"loss_t_coeff_is_nan: {loss_t_coeff_is_nan}")

        loss_t_coeff = loss_t_coeff.detach().cpu().numpy()
        coeffs = coeff.detach().cpu().numpy()
        plt.figure(figsize=(10, 6))
        plt.scatter(ts, coeffs, c=ts, cmap='magma')
        plt.ylabel('coeff')
        plt.xlabel('t')
        plt.title(f"Coeff vs t, rate={self.rate}")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, f"{name}_coeff.png"), dpi=300, bbox_inches='tight')
        wandb.log({f"{name}_coeff_step={self.global_step}": wandb.Image(plt.gcf())})
        plt.close()


        losses = loss.detach().cpu().numpy()

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(ts, losses, c=ts, cmap='magma')
        plt.colorbar(scatter, label='t')
        plt.ylabel('loss')
        plt.xlabel('t')
        plt.title(f"Loss vs t, no coeff, rate={self.rate}")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path,f"{name}.png"), dpi=300, bbox_inches='tight')
        wandb.log({f"{name}_loss_no_coeff_step={self.global_step}": wandb.Image(plt.gcf())})
        plt.close()

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(ts, loss_t_coeff, c=ts, cmap='magma')
        plt.colorbar(scatter, label='t')
        plt.ylabel('loss')
        plt.xlabel('t')
        plt.title(f"Loss vs t, w coeff, rate={self.rate}")
        plt.grid(True, alpha=0.3)
        # plt.xlim(0, 0.2)
        plt.savefig(os.path.join(save_path,f"{name}.png"), dpi=300, bbox_inches='tight')
        wandb.log({f"{name}_loss_with_coeff_step={self.global_step}": wandb.Image(plt.gcf())})
        plt.close()
        
    def pre_configure_model(self, dataloader, val_dataloader=None):
        """
        Assumes linear schedule_condition. No other schedule_condition is currently implemented
        """
        if dataloader is not None:
            self.calc_p0(dataloader)         
        self.gamma = lambda t: t 
        self.gamma_prime = lambda t: 1
    
    def get_stationary(self):
        stationary = torch.ones(self.num_classes)
        stationary = stationary / self.num_classes
        return stationary
    
    @staticmethod
    def stable_1m_exp(x):
        return torch.exp(-x) * torch.expm1(x)

    def sample_point(self, x):
        """
        Gaussian diffusion - sampling, algorithm 2
        t ~ Unif(0, 1)
        x_t = e^{- tau_t} * emb(x_0) + sqrt(1 - e^{-2 tau_t}) * N(0, I)
        """
        with torch.no_grad():
            B, D = x.shape[0], x.shape[1]
            x = convert_to_probs(x, self.num_classes).float()  # [B, L, C] where C is one-hot encoded
            ts = torch.rand(x.shape[0], device=x.device).detach() * self.t_max  # t ~ Unif(0, t_max), t_max ~= 1
            tau_t = self.tau(ts).view(len(ts), 1, 1)

            N = torch.randn((B, D, self.num_classes), device=x.device)  # [B, D, C]

            x_t = torch.exp(-tau_t) * x + torch.sqrt(self.stable_1m_exp(2 * tau_t)) * N
            
            return ts, x_t

    def compute_loss(self, x, predicted_x0_logits, t):
        pred_x0 = torch.nn.functional.softmax(predicted_x0_logits, dim=-1)
        x0 = convert_to_probs(x, self.num_classes).float()
        e_neg_two_tau_t = torch.exp(-2 * self.tau(t))
        coeff = self.beta(t) * e_neg_two_tau_t / torch.square(self.stable_1m_exp(2 * self.tau(t)))
        norm = torch.sum(torch.square(x0 - pred_x0), dim=-1)

        loss = coeff.unsqueeze(-1) * norm * self.t_max
        return loss

    def compute_loss_advanced(self, x, predicted_x0_logits, t):
        pred_x0 = torch.nn.functional.softmax(predicted_x0_logits, dim=-1)
        x0 = convert_to_probs(x, self.num_classes).float()
        e_neg_two_tau_t = torch.exp(-2 * self.tau(t))
        coeff = self.beta(t) * e_neg_two_tau_t / torch.square(1 - e_neg_two_tau_t)
        norm = torch.sum(torch.square(x0 - pred_x0), dim=-1)
        loss_wo_coeff = norm * self.t_max
        return loss_wo_coeff, coeff.unsqueeze(-1)


    def p_x_t_given_x_0(self, x_t, ts):
        """
        Computes log prob of x_t given x_0, t

        x_t: [B, D, C]
        ts: [B]
        """
        with torch.no_grad():
            B, _, C = x_t.shape
            tau_t = self.tau(ts).view(B, 1, 1)

            exp_neg_tau_t = torch.exp(-tau_t)
            exp_neg_two_tau_t = torch.exp(-2 * tau_t)
            sigma_sq = self.stable_1m_exp(2 * tau_t)

            x_t_norm_sq = torch.sum(x_t ** 2, dim=-1, keepdim=True)
            mu_norm_sq = exp_neg_two_tau_t   

            x_t_dot_mu = exp_neg_tau_t * x_t   # mu is one-hot - dot-prod
            mahalanobis = (x_t_norm_sq + mu_norm_sq - 2 * x_t_dot_mu) / sigma_sq

            log_prob = -0.5 * (
                C * torch.log(2 * torch.tensor(math.pi, device=ts.device) * sigma_sq) +
                mahalanobis
            )
            return log_prob
    
    def forward(self, x, attn_mask=None, zeta=None, **kwargs):
        """
        x: sequences described by letter indices. shape: batch_size x max(L)
        """
        t, x_t = self.sample_point(x)  # Algorithm 2.1, 2.3
        p_x_t_given_x_0 = self.p_x_t_given_x_0(x_t=x_t, ts=t)
 
        if self.ssp:
            x_t = torch.log(self.p0).to(x.device) + p_x_t_given_x_0 # applying Bayes rule
            # x_t[:,:,-1] -= 100 # padding token
            x_t = torch.nn.functional.softmax(x_t, dim=-1)
            input = x_t
            time = torch.ones_like(t)
        else:
            input, time = x_t, t
        
        NN_predicted_x0_logits = self.model_predict(input.float(), time.float(), attn_mask).to(torch.float32)  # Algorithm 2.5
        predicted_x0_logits = NN_predicted_x0_logits + p_x_t_given_x_0   # hollow parameterization

        loss = self.compute_loss(x, predicted_x0_logits, t)

        if attn_mask is not None:
            loss = loss * attn_mask

        vb_loss = loss.mean() * self.t_max 
        vb_loss_log = loss.mean(dim = -1) * self.t_max

        if attn_mask is not None:
            vb_loss = vb_loss / attn_mask.mean() 
            vb_loss_log = vb_loss_log / attn_mask.mean() 


        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(predicted_x0_logits, x) 
        if attn_mask is not None:
            ce_loss = (ce_loss * attn_mask.flatten()).sum() / attn_mask.sum()
        else:
            ce_loss = ce_loss.mean()
        
         # wandb log the loss for each batch torch.mean(loss, dim=-1), also log the loss for each t. 
        t_list = t.detach().cpu().tolist() if t.requires_grad else t.cpu().tolist()
        loss_list =vb_loss_log.detach().cpu().tolist() # loss is nxd. average over d
        for i in range(len(t_list)):
            wandb.log({"t_iter":float(t_list[i]), "loss_iter":float(loss_list[i])})
        
        return vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }
        
    def sample_step(self, x_t, x_0_pred, t, delta_t):
        tau_t = self.tau(t).view(-1, 1, 1)
        delta_tau = self.tau(t).view(-1, 1, 1) - self.tau(t-delta_t).view(-1, 1, 1)
        tau_t_min_eps = tau_t - delta_tau

        mu_1 = torch.exp(-tau_t_min_eps) * x_0_pred
        mu_2 = torch.exp(delta_tau) * x_t

        var1 = self.stable_1m_exp(2 * tau_t_min_eps)
        var2 = torch.expm1(2 * delta_tau)

        denom = var1 + var2

        mu = (mu_1 * var2 + mu_2 * var1) / denom
        var = var1 * var2 / denom

        x_sample = mu + torch.sqrt(var) * torch.randn(size=x_t.shape, device=x_t.device)

        return x_sample
        
    def noise(self, x_t, t, delta_t):
        tau_t = self.tau(t).view(-1, 1, 1)
        delta_tau = self.tau(t+delta_t).view(-1, 1, 1) - self.tau(t).view(-1, 1, 1)
        
        mu = torch.exp(-delta_tau) * x_t
        var = self.stable_1m_exp(-2 * delta_tau)
        
        return mu + torch.sqrt(var) * torch.randn(size=x_t.shape, device=x_t.device)

    def sample(self, B, D, delta_t, device='cpu', max_D=None, corrector_steps=0, **kwargs):
        max_D = D if max_D is None else max_D
        attn_mask = torch.ones((B, max_D), dtype=torch.bool).to(device)
        attn_mask[:, D:] = 0
        
        C = self.num_classes
        x_t = torch.randn(size=(B, max_D, C), device=device)

        with torch.no_grad():
            t = self.t_max
            while t > delta_t:  # t > 0 - silent error when t - delta_t <= 0
                t_tensor = torch.ones(size=(B,), device=device) * t
                delta_t_tensor = torch.ones(size=(B,), device=device) * delta_t
                density = self.p_x_t_given_x_0(x_t, t_tensor)

                if self.ssp:
                    input = torch.log(self.p0).to(x_t.device) + density
                    input = torch.nn.functional.softmax(input, dim=-1)
                    input_time = torch.ones_like(t_tensor)
                else:
                    input = x_t
                    input_time = t_tensor
                pred_x0 = self.model_predict(input, input_time, attn_mask) + density
                pred_x0 = torch.nn.functional.softmax(pred_x0, dim=-1)

                x_sample = self.sample_step(x_t, pred_x0, t_tensor, delta_t_tensor)
                
                for _ in range(corrector_steps):
                    # add noise
                    x_t_noisy = self.noise(x_t, delta_t_tensor, self.zeta)
                    # denoise
                    x_t = self.sample_step(x_t_noisy, pred_x0, t_tensor, delta_t_tensor)
                    
                x_t = x_sample
                t -= delta_t

            x_t = torch.argmax(x_t, dim=-1)

        return x_t[:, :D]
    
