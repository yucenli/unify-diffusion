import torch
import torch.nn.functional as F
from tqdm import tqdm
from time import time as current_time
import wandb
import logging

from .utils.utils import (
    convert_to_probs,
    simulate_ancestral_process_mixed,
    PiecewiseExponential
)
from .utils.utils_F_G import (
    compute_series_torch,
    compute_series_precise_bad,
)
from .trainer import ContinuousTimeDiffusion

logger = logging.getLogger(__name__)


class SimplicialDiffusion(ContinuousTimeDiffusion):
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes,
        forward_kwargs={"type":"uniform", 
                        "ssp":"false"},
        schedule_type="cos",
        zeta=1,
        psi=0.5,
        logistic_pars=False,
        recompute_precise=False,
        **kwargs
    ):
        # Precalculate betas, define model_predict, p_sample
        super().__init__(x0_model_class, nn_params, num_classes, schedule_type, logistic_pars, **kwargs)
        self.save_hyperparameters(ignore=['x0_model_class'])
        self.eps=1e-6
        assert zeta >= 1
        self.zeta = zeta
        self.num_classes = num_classes

        self.psi = psi
        self.recompute_precise = recompute_precise
        self.total_small_error = 0
        self.ssp = forward_kwargs['ssp']
        self._last_intermediates = None  # for debugging
        
    def pre_configure_model(self, train_dataloader, val_dataloader=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        if self.num_classes == 4:
            self.dilation = PiecewiseExponential(slopes=[8, -2], intercepts=[0, 2])
        else:
            slopes=[20, 10, -15]
            intercepts=[1, 2, 6]
            self.dilation = PiecewiseExponential(slopes=slopes, intercepts=intercepts)
        self.compute_tau = self.dilation.compute_tau
        self.compute_beta = self.dilation.compute_beta
    
    def get_stationary(self, device=None):
        stationary = torch.ones(self.num_classes) / self.num_classes
        if device is not None:
            stationary = stationary.to(device)
        return stationary

    def kingmans_coalescent_A(self, tau, D):
        taus = tau.unsqueeze(1).repeat(1, D)
        out = torch.zeros_like(taus)
        for i in range(0, len(taus), 32):
            out[i:i+32] = simulate_ancestral_process_mixed(taus[i:i+32], self.psi)
            
        return out

    def griffiths_approximation(self, tau, D):
        '''
        Theorem 1 from P. A. JENKINS & D. SPANO WRIGHT-FISHER DIFFUSION
        returns A_inf^theta(t) values in shape (len(t) x D)
        '''
        beta = 0.5 * (self.psi - 1) * tau
        
        epsilon = 1e-10 # small threshold for numerical stability
        beta_mask_zero = torch.abs(beta) < epsilon
        
        eta = torch.zeros_like(beta)
        sigma_sq = torch.zeros_like(beta)
        
        # For beta != 0
        beta_nonzero = ~beta_mask_zero
        if beta_nonzero.any():
            beta_nz = beta[beta_nonzero]
            # eta_nz = beta_nz / (torch.exp(beta_nz) - 1)
            eta_nz = beta_nz / torch.expm1(beta_nz)
            eta[beta_nonzero] = eta_nz

            bracket_term = 1 + eta_nz/(eta_nz + beta_nz) - 2*eta_nz
            sigma_sq[beta_nonzero] = (2 * eta_nz / tau[beta_nonzero]) * \
                                ((eta_nz + beta_nz)**2) * \
                                bracket_term / (beta_nz**2)
            
        
        # For beta == 0
        if beta_mask_zero.any():
            eta[beta_mask_zero] = 1
            sigma_sq[beta_mask_zero] = 2 / (3 * tau[beta_mask_zero])
        
        mu = 2 * eta / tau
        
        # Sample from normal and round
        samples = mu.unsqueeze(-1) + torch.sqrt(sigma_sq).unsqueeze(-1) * torch.randn((len(tau), D)).to(tau.device)
        samples = torch.clamp(torch.round(samples), min=0)
        
        # if nan return 100
        samples = torch.where(torch.isnan(samples), torch.tensor(100.0).to(samples.device), samples)
        return samples

    def compute_dirichlet_params(self, x_0, m):
        B, D = x_0.shape[:2]
        pi = self.get_stationary().to(x_0.device)
        stationary = (self.psi * pi).unsqueeze(0).unsqueeze(0).expand(B, D, -1)
        x_0_probs = convert_to_probs(x_0, self.num_classes).to(x_0.device)
        m_x0 = m.unsqueeze(-1) * x_0_probs # B x D x num_classes
        concentration = stationary + m_x0
        return concentration

    def sample_point(self, x_0, attn_mask=None, rand_shape=None):
        # The procedure is 1) simulate m, 2) draw X_t\sim Dirichlet(\theta\pi+mX_0)
        with torch.no_grad():
            B, D = x_0.shape[:2]
            ts = torch.rand(B, device=x_0.device, dtype=torch.float64).detach() * self.t_max # t is uniformly distributed. Each dimension d has the same t 
            taus = self.compute_tau(ts)

            small_mask = taus < 0.1 
            taus_small = taus[small_mask]  # Griffith's
            taus_large = taus[~small_mask]  # Kingman's
            
            m = torch.zeros((B, D), dtype=torch.float64).to(x_0.device)
            if small_mask.any():
                m[small_mask] = self.griffiths_approximation(taus_small, D).double().to(x_0.device)
            if (~small_mask).any():
                m[~small_mask] = self.kingmans_coalescent_A(taus_large, D).double().to(x_0.device)
            
            concentration = self.compute_dirichlet_params(x_0, m)
            x_t = torch.distributions.Dirichlet(concentration).sample()
            
            return ts, taus, x_t, m
           
    def compute_loss(self, x_0, pred_x0, t, tau, x_t, F_vec=None, G_vec=None, F_bad=None, G_bad=None):
        B, D, C = x_t.shape
        small_mask = tau < 0.05
        small_error_rate = 0.0
        
        loss = torch.zeros((B, D), device=x_0.device)
        if small_mask.any():
            # check x_t argmax is x0
            x_0_small = x_0[small_mask]
            x_t_small = x_t[small_mask]
            
            largest_indices = torch.argmax(x_t_small, dim=-1)
            small_error = (largest_indices != x_0_small).sum().item()
            if small_error != 0:
                logging.warning(f"{small_error} / {x_0_small.numel()} errors in small tau regime")
            self.total_small_error += small_error
            small_error_rate = small_error / x_0_small.numel()
        
        if (~small_mask).any():
            tau_large = tau[~small_mask]
            x_t_large = x_t[~small_mask]
            x_0_large = x_0[~small_mask]
            pred_x0_large = pred_x0[~small_mask]
            B_large = x_t_large.shape[0]
            
            num = (torch.exp(-self.psi * tau_large / 2) * (self.psi + 1)).unsqueeze(-1).unsqueeze(-1).expand(-1, D, C) # shape B x D x C
            pi = self.get_stationary(device=x_0.device).unsqueeze(0).unsqueeze(0).expand(-1, D, C) # shape B x D x C
            if G_vec is None or F_vec is None:
                bs = torch.arange(C, device=x_0.device).unsqueeze(0).unsqueeze(0).expand(B_large, D, C) # shape B x D x C
                G_vec, G_bad = self.compute_series('G', tau_large, bs, x_t_large) # shape B x D x C
                F_vec, F_bad = self.compute_series('F', tau_large, bs, x_t_large) # shape B x D x C
                bad_cond = G_bad | F_bad
            else:
                F_vec = F_vec[~small_mask]
                G_vec = G_vec[~small_mask]
                F_bad = F_bad[~small_mask]
                G_bad = G_bad[~small_mask]
            
            onehot_x0 = F.one_hot(x_0_large, num_classes=C).to(torch.float64) # shape B x D x C
            true_F_term = onehot_x0 * F_vec # shape B x D x C
            true_G_sum = (onehot_x0 * G_vec).sum(-1, keepdim=True) # shape B x D x 1
            true_score = num / pi * true_F_term / true_G_sum
            
            pred_F_term = pred_x0_large * F_vec # shape B x D x C
            pred_G_sum = (pred_x0_large * G_vec).sum(-1, keepdim=True) # shape B x D x 1
            pred_score = num / pi * pred_F_term / pred_G_sum
            
            score_diff = (true_score - pred_score).to(torch.float32)
            diag_loss = (x_t_large * (score_diff ** 2)).sum(-1)
            outer = (x_t_large * score_diff).sum(-1)
            outer_loss = outer * outer
            
            beta = self.compute_beta(t[~small_mask]).unsqueeze(-1)
            loss_large = beta / 2 * (diag_loss - outer_loss)
            loss[~small_mask] = loss_large.float()
            
            large_loss = (loss_large > 10.0).unsqueeze(-1)
            if self.recompute_precise and (bad_cond & large_loss).any():
                logging.warning(f"{(bad_cond & large_loss).sum().item()} / {B_large * D} large losses in large tau regime due to series divergence")
                retry_G = compute_series_precise_bad('G', tau_large, bs, x_t_large, 80, bad_cond) 
                retry_F = compute_series_precise_bad('F', tau_large, bs, x_t_large, 80, bad_cond)
                
                G_vec = torch.where(bad_cond, retry_G, G_vec)
                F_vec = torch.where(bad_cond, retry_F, F_vec)
                
                onehot_x0 = F.one_hot(x_0_large, num_classes=C).to(torch.float64) # shape B x D x C
                true_F_term = onehot_x0 * F_vec # shape B x D x C
                true_G_sum = (onehot_x0 * G_vec).sum(-1, keepdim=True) # shape B x D x 1
                true_score = num / pi * true_F_term / true_G_sum
                
                pred_F_term = pred_x0_large * F_vec # shape B x D x C
                pred_G_sum = (pred_x0_large * G_vec).sum(-1, keepdim=True) # shape B x D x 1
                pred_score = num / pi * pred_F_term / pred_G_sum
                
                score_diff = (true_score - pred_score).to(torch.float32)
                diag_loss = (x_t_large * (score_diff ** 2)).sum(-1)
                outer = (x_t_large * score_diff).sum(-1)
                outer_loss = outer * outer
                
                beta = self.compute_beta(t[~small_mask]).unsqueeze(-1)
                loss_large = beta / 2 * (diag_loss - outer_loss)
                loss[~small_mask] = loss_large.float()

        return loss * self.t_max, small_error_rate, (F_vec, G_vec, F_bad, G_bad)
    
    def compute_vec(self, tau, x_t, series='G'):
        B, D, C = x_t.shape
        bs = torch.arange(C, device=x_t.device).unsqueeze(0).unsqueeze(0).expand(B, D, C) # shape B x D x C
        vec, bad = self.compute_series(series, tau, bs, x_t) # shape B x D x C
        return vec
    
    def compute_series(self, series, t, x_0, x_t, max_terms=80, tol=1e-6):
        return compute_series_torch(series, t, x_0, x_t, max_terms)
    
    def forward(self, x, attn_mask=None, extra=None):
        """
        x: sequences described by letter indices. shape: batch_size x max(L)
        """
        # cached datasets
        if extra is not None:
            t, tau, m, x_t, F_psi, G_psi, F_bad, G_bad = extra
            attn_mask = attn_mask.bool()
        else:
            F_psi, G_psi, F_bad, G_bad = None, None, None, None
            t, tau, x_t, m = self.sample_point(x, attn_mask)
            if torch.isnan(x_t[tau > 0.05]).any():
                logger.warning("nan in x_t")
             
        if self.ssp:
            # p(x_t | x_0, t) at position b = Dirichlet() * G_psi(x_0)
            # normalized = G_psi(position b) / sum_b' G_psi(position b')
            if G_psi is None:
                G_psi = self.compute_vec(tau, x_t, series='G')
            input = G_psi / G_psi.sum(-1, keepdim=True)  # B x D x C
            time = torch.ones_like(t)
            # set nan to zero
            input = torch.where(torch.isnan(input), torch.zeros_like(input), input)
        else:
            input, time = x_t, t

        predicted_x0_logits = self.model_predict(input.float(), time.float(), attn_mask).to(torch.float64)
        predicted_x0 = torch.nn.functional.softmax(predicted_x0_logits, dim=-1)
        loss_start_time = current_time() 
        loss, small_error_rate, series_info = self.compute_loss(x, predicted_x0, t, tau, x_t, F_psi, G_psi, F_bad, G_bad)
        loss_time = current_time() - loss_start_time
        wandb.log({"loss_time": loss_time})   
        F_vec, G_vec, F_bad, G_bad = series_info
        
        self._last_intermediates = {
            "t": t.detach(),
            "tau": tau.detach(),
            "x_t": x_t.detach(),
            "x_0": x.detach(),
            "m": m.detach(),
            "predicted_x0": predicted_x0.detach(),
            "loss": loss.detach(),
            "F_vec": F_vec.detach(),
            "G_vec": G_vec.detach(),
            "F_bad": F_bad.detach(),
            "G_bad": G_bad.detach()
        }
            
        if attn_mask is not None:
            loss = loss[attn_mask.bool()]
        
        if torch.isnan(loss).any():
            logger.warning("nan in loss")

        # wandb log the loss for each batch torch.mean(loss, dim=-1), also log the loss for each t. t is nxd 
        vb_loss = loss.mean()
        wandb.log({
            "total_small_error": self.total_small_error,
            "small_error_rate": small_error_rate,
            "loss": vb_loss
        })
        
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(predicted_x0_logits, x) 
        if attn_mask is not None:
            ce_loss = (ce_loss * attn_mask.flatten()).sum() / attn_mask.sum()
        else:
            ce_loss = ce_loss.mean()

        return vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }
        
    def score(self, pred_x0, x_t, tau):
        B, D, C = x_t.shape
        
        diri_grad = 0#(self.psi * self.get_stationary(device=pred_x0.device) - 1) / x_t
        
        F_psi_vec = self.compute_vec(tau, x_t, 'F')  # B x D x C
        G_psi_vec = self.compute_vec(tau, x_t, 'G')  # B x D x C

        pred_F_term = pred_x0 * F_psi_vec  # B x D x C
        pred_G_sum = (pred_x0 * G_psi_vec).sum(-1, keepdim=True)  # B x D x 1
                    
        num = (torch.exp(-self.psi * tau / 2) * (self.psi + 1))[:, None, None].expand(-1, D, C) # shape B x D x C
        pi = self.get_stationary(device=pred_x0.device)[None, None, :].expand(-1, D, C) # shape B x D x C
        return diri_grad + num / pi * pred_F_term / pred_G_sum
    
    def compute_t_from_tau(self, tau):
        a, b = torch.tensor(0.0).to(tau), torch.tensor(1.0).to(tau)
        while b - a > 1e-4:
            mid = (a + b) / 2
            if self.compute_tau(mid) < tau:
                a = mid
            else:
                b = mid
        return (a + b) / 2
        
    def sample_step(self, x_t, t, delta_t, score_fn, eps=1e-5, forward=False):
        tau_t = self.compute_tau(t)
        if forward:
            tau_t_2 = self.compute_tau(t + delta_t)
            d_tau = tau_t_2 - tau_t
        else:
            tau_t_2 = self.compute_tau(t - delta_t)
            d_tau = tau_t - tau_t_2
        return self.sample_step_tau(x_t, tau_t, d_tau, score_fn, eps, t=t, forward=forward)
        
    def sample_step_tau(self, x_t, tau_t, d_tau, score_fn, eps=1e-5, t=None, forward=False):
        if t is None:
            t = self.compute_t_from_tau(tau_t[0].item())
            t = torch.ones_like(tau_t) * t
            print(f"t is {t[0]} at tau {tau_t[0]}")
        d_tau = d_tau[:, None, None]
        pi = self.get_stationary(device=x_t.device)
        z_t = x_t
    
        # drift
        C = self.num_classes
        term1 = (self.psi / 2) * (pi - z_t) - C * (1/C - z_t)
        if forward:
            score = 0
        else:
            score = score_fn(z_t, t, tau_t)
        term2 = z_t * score - z_t * (z_t * score).sum(-1, keepdim=True)
        drift = term1 - term2
        
        # noise
        gauss = torch.randn_like(z_t)
        noise = torch.sqrt(z_t) * (gauss - torch.sqrt(z_t) * (torch.sqrt(z_t) * gauss).sum(-1).unsqueeze(-1))
        
        z_t = z_t - drift * d_tau + noise * torch.sqrt(d_tau)
        
        # projection to simplex
        z_t = torch.clamp(z_t, eps, 1-eps)
        z_t = z_t / z_t.sum(-1, keepdim=True)
        
        return z_t
    
    def sample(self, B, D, delta_t, device='cpu', max_D=None, corrector_steps=0, temp=1.0):
        max_D = D if max_D is None else max_D
        attn_mask = torch.ones((B, max_D), dtype=torch.bool).to(device)
        attn_mask[:, D:] = 0

        def unconditional_score(x_t, t_tensor, tau):
            if self.ssp:
                G_psi = self.compute_vec(tau, x_t, series='G')
                input = G_psi / G_psi.sum(-1, keepdim=True)  # B x D x C
                input = torch.where(torch.isnan(input), torch.zeros_like(input), input)
                time = torch.ones_like(t_tensor)
            else:
                input, time = x_t, t_tensor

            with torch.no_grad():
                pred_x0 = self.model_predict(input.float(), time.float(), attn_mask)
                pred_x0 = torch.nn.functional.softmax(pred_x0 / temp, dim=-1)
                score = self.score(pred_x0, x_t, tau)
            return score
        
        return self._sample(B, max_D, delta_t, unconditional_score, device=device, corrector_steps=corrector_steps)[:, :D]
    
    def p_x_t_given_x_0(self, x_t, tau):
        diri_dist = torch.distributions.Dirichlet(self.psi * self.get_stationary(device=x_t.device))
        diri_log_prob = diri_dist.log_prob(x_t).unsqueeze(-1)  # B x D x 1
        G_vec = self.compute_vec(tau, x_t, 'G')  # B x D x C
        G_vec[G_vec <= 0] = 1e-10
        return G_vec.log() + diri_log_prob
        
    def conditional_sample(self, B, D, delta_t, guidance_model, target, device='cpu'):
        def conditional_score(x_t, t_tensor, tau, target=target):
            x_t = x_t.clone().detach().requires_grad_(True)
            
            attn_mask = torch.ones((B, D), dtype=torch.bool).to(device)
            
            pred_x0 = self.model_predict(x_t.float(), t_tensor.float(), attn_mask)
            pred_x0_softmax = torch.nn.functional.softmax(pred_x0, dim=-1)
            
            hollow_pred_x0 = pred_x0 + self.p_x_t_given_x_0(x_t, tau)
            pred_x0_density_softmax = torch.nn.functional.softmax(hollow_pred_x0, dim=-1)
            
            guidance_log_prob = guidance_model(pred_x0_density_softmax.float(), target)
            grad = torch.autograd.grad(guidance_log_prob.sum(), x_t)[0].detach()
            
            with torch.no_grad():
                score = self.score(pred_x0_softmax, x_t, tau)
            
            return score + grad
        
        return self._sample(B, D, delta_t, conditional_score, device=device)

    def _sample(self, B, D, delta_t, score_fn, device='cpu', corrector_steps=0):
        diri_dist = torch.distributions.Dirichlet(self.psi * self.get_stationary(device=device))
        x_t = diri_dist.sample((B, D)).to(device)

        t = self.t_max
        delta_t_tensor = torch.ones(size=(B,), device=device) * delta_t

        num_steps = int(t // delta_t)
        for _ in tqdm(range(num_steps), desc="Sampling", leave=False, disable=False):
            t_tensor = torch.ones(size=(B,), device=device) * t
            x_t = self.sample_step(x_t, t_tensor, delta_t_tensor, score_fn, forward=False)
            
            for step in range(corrector_steps):
                t_denoised = t_tensor - delta_t
                # add noise back up to previous level
                x_t_noisy = self.sample_step(x_t, t_denoised, delta_t_tensor, score_fn, forward=True)
                # denoise again
                x_t = self.sample_step(x_t_noisy, t_tensor, delta_t_tensor, score_fn, forward=False)
            t -= delta_t
            
        x_t = torch.argmax(x_t, dim=-1)

        return x_t
    
    def _sample_tau(self, B, D, delta_tau, score_fn, device='cpu', corrector_steps=0):
        diri_dist = torch.distributions.Dirichlet(self.psi * self.get_stationary(device=device))
        x_t = diri_dist.sample((B, D)).to(device)

        t = self.t_max
        tau = self.compute_tau(torch.tensor([t], device=device)).item()
        delta_tau_tensor = torch.ones(size=(B,), device=device) * delta_tau

        num_steps = int(tau // delta_tau)
        for _step in tqdm(range(num_steps), desc="Sampling", leave=False, disable=False):
            tau_tensor = torch.ones(size=(B,), device=device) * tau
            x_t = self.sample_step_tau(x_t, tau_tensor, delta_tau_tensor, score_fn, forward=False)
            
            for step in range(corrector_steps):
                tau_denoised = tau_tensor - delta_tau
                # add noise back up to previous level
                x_t_noisy = self.sample_step_tau(x_t, tau_denoised, -delta_tau_tensor, score_fn, forward=True)
                # denoise again
                x_t = self.sample_step_tau(x_t_noisy, tau_tensor, delta_tau_tensor, score_fn, forward=False)
            tau -= delta_tau_tensor
            
        x_t = torch.argmax(x_t, dim=-1)

        return x_t