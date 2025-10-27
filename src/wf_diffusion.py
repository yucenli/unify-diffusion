import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
import wandb
import random

from .utils.utils import kls, convert_to_probs, get_inf_gen, sample_index_S, multinomial_numba
from .trainer import ContinuousTimeDiffusion


class WFDiffusion(ContinuousTimeDiffusion):
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
        # Precalculate betas, define model_predict, p_sample
        super().__init__(x0_model_class, nn_params, num_classes, schedule_type, logistic_pars, **kwargs)
        self.save_hyperparameters(ignore=['x0_model_class'])
        self.eps=1e-6
        self.ssp = ssp
        self.num_classes = num_classes
        
        L = get_inf_gen(forward_kwargs, num_classes)
        self.L = L
        eigenvalues, eigenvectors = torch.linalg.eig(L.to(torch.float64))
        
        # sort by eigenval magnitude
        indices = torch.argsort(torch.abs(eigenvalues))
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        eigenvectors_inv = torch.linalg.inv(eigenvectors)
        
        self.register_buffer("eigenvalues", eigenvalues)
        self.register_buffer("eigenvectors", eigenvectors)
        self.register_buffer("eigenvectors_inv", eigenvectors_inv)

        self.tau, self.beta = self.calculate_tau()


    def calculate_tau(self):
        # find magnitude second smallest eigenval
        eigval_1 = torch.abs(self.eigenvalues[1])
        tau = lambda t, zeta=1.: 1/eigval_1 * torch.log(math.sqrt(zeta) * torch.sqrt((self.gamma(t)+1/zeta) / (1 - self.gamma(t))))
        tau_prime = lambda t, zeta=1.:  (zeta + 1) / (eigval_1 * 2 * (zeta * self.gamma(t) + 1)*(1-self.gamma(t))) * self.gamma_prime(t)
     
        return tau, tau_prime
        
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

    def get_trans_mats_mvp(self, Smk, v):
        """ v is a ...c matrix where K is cd, and Smk is ... """
        dv = v.to(dtype=self.eigenvectors.dtype).reshape(-1, v.shape[-1])
        diag = self.eigenvalues ** F.relu(Smk.flatten()[..., None])
        dv = dv @ self.eigenvectors
        dv = dv * diag
        dv = dv @ self.eigenvectors_inv
        return F.relu(dv.double()).to(torch.float32).reshape(v.shape)
    
    # def get_kl_t1(self, x):
    #     # sample S
    #     t = self.t_max * torch.ones(x.shape[0], device=x.device)
    #     S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
    #     S = S.swapaxes(0, 1).reshape(*x.shape).long()
    #     softmaxed = convert_to_probs(x, self.num_classes)  # bs, ..., num_classes
    #     trans = self.get_trans_mats_mvp(S, softmaxed)
    #     x_1 = torch.log(trans+self.eps)
    #     kl = kls(x_1, torch.log(self.get_stationary() + self.eps))
    #     return kl.mean()

    def rate_exponential(self, t, zeta):
        # returns rate exponential, of shape batch_size x B x B
        tau_t = self.tau(t, zeta).view(-1, 1)  # shape: (batch_size, 1)
        scaled_eigenvals = tau_t * self.eigenvalues  # broadcast over eigenvalues
        eigenval_exp = torch.diag_embed(torch.exp(scaled_eigenvals))
        matrix_exponential = self.eigenvectors @ eigenval_exp @ self.eigenvectors_inv
        matrix_exponential = torch.real(matrix_exponential) # will be real, because L and tau are real
        matrix_exponential = torch.where((matrix_exponential < 0) & (matrix_exponential >= -self.eps) , # clamp negative values that are small due to numerical errors
                                0, 
                                matrix_exponential)
        return matrix_exponential

    def compute_probabilities(self, x, t, zeta):
        '''
        Returns array of  X^T e^(τ(t)L) with shape (batch_size x D x B)
        '''
        x = convert_to_probs(x, self.num_classes).double()
        matrix_exponential = self.rate_exponential(t, zeta).to(x.device)
        matrix_exponential = matrix_exponential.unsqueeze(1).expand(-1,x.shape[1],-1,-1) # expand across D to be n x D x B x B
        
        # X^T @ e^(tau * L) 
        # x is shape n x D x 1 x B. e^(tau L) is shape n x D x B x B
        probabilities = torch.matmul(x.unsqueeze(-1).transpose(3,2), matrix_exponential)
        probabilities = probabilities.squeeze(2)

        return probabilities
    
    def sample_point_(self, x, zeta, attn_mask=None, rand_shape=None):   
        with torch.no_grad():
            batch_size = x.shape[0]
            D = x.shape[1]
            ts = torch.rand(x.shape[0], device=x.device).detach() * self.t_max # t is uniformly distributed. Each dimension d has the same t 
            
            probabilities = self.compute_probabilities(x, ts, zeta=zeta)
            probabilities = torch.clamp(probabilities, max=1.0) # adjust values that are over 1 due to numerical stability
            probabilities = torch.flatten(probabilities, start_dim=0, end_dim=1)
            
            S = np.full((batch_size * D,), zeta) # the counts for each D in each batch
            counts = torch.from_numpy(multinomial_numba(p=probabilities.detach().cpu().numpy(), S=S, num_classes=self.num_classes)).to(x.device)
            counts = counts.view(batch_size, D, self.num_classes)

            # rescale the counts Xt = √(1 − γt + γtζ)( Xt ζ − π)
            count_distribution = counts / zeta
            pi = self.get_stationary().to(x.device)
            x_t = torch.sqrt(1-self.gamma(ts) + self.gamma(ts) * zeta).view(-1,1,1) * (count_distribution - pi)
            return ts, x_t, counts
   

    def compute_loss(self, x, predicted_x0_logits, t, x_t_counts, zeta):
        p = self.compute_probabilities(x, t, zeta=zeta)
        q = self.compute_probabilities(predicted_x0_logits, t, zeta=zeta)

        sum = torch.zeros_like(x, dtype=torch.float32)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i == j:
                    continue
                rate = self.L[j, i]
                l_p = (p[:,:,j] + self.eps)/ (p[:,:,i] + self.eps)
                l_q = (q[:,:,j] + self.eps)/ (q[:,:,i] + self.eps)
                D_kl = l_p * torch.log(l_p/l_q) + l_q - l_p # D_lk is (p_b2 / p_b1) * log ((p_b2/p_b1)/(q_b2/q_b1)) + (q_b2/q_b1) - (p_b2/p_b1)

                out = rate * self.beta(t, zeta).unsqueeze(-1) * x_t_counts[:,:,i] * D_kl
                sum +=out

        return sum


    def p_x_t_given_x_0(self, x_t_counts, ts, zeta):
        """
        x_t_counts is a tensor of the counts of each letter. Shape n x D x B
        computes log p(x_t^d|x_0^d,s_t^d)
        by computing sum_{alphabet} X_t * log( X_0^T e^(τ(t)L))
        """
        all_letters = torch.arange(self.num_classes)
        x_0 = all_letters.view(1,-1).expand(len(ts), -1).to(x_t_counts.device) 
        x_0 = convert_to_probs(x_0, self.num_classes).to(torch.float32) # n x B x B

        matrix_exponential = self.rate_exponential(ts, zeta).to(x_t_counts.device).to(torch.float32) # n x B x B

        # note: x_0 = I, so x_0 @ e^(τ(t)L) = e^(τ(t)L)
        scaled_by_counts = x_t_counts.to(torch.float32).unsqueeze(-1) * torch.log(matrix_exponential).unsqueeze(1).transpose(-2,-1) # n x D x B x 1 time (nx1xBxB)
        scaled_by_counts = torch.nan_to_num(scaled_by_counts, nan=0) # scaled values are nan when matrix exp is zero, but if counts are zero in these spots, the likelihood of x_t should not necessarily all be -inf
        likelihood = torch.sum(scaled_by_counts, dim=2)
        return likelihood 
    
    def forward(self, x, attn_mask=None, zeta=None, **kwargs):
        """
        x: sequences described by letter indices. shape: batch_size x max(L)
        """
        t, x_t, x_t_counts = self.sample_point(x, zeta, attn_mask)
        
        if self.ssp:
            p_x_t_given_x_0 = self.p_x_t_given_x_0(x_t_counts=x_t_counts, ts=t, zeta=zeta)
            x_t = torch.log(self.p0).to(x.device) + p_x_t_given_x_0 # applying Bayes rule
            x_t[:,:,-1] -= 100 # padding token
            x_t = torch.nn.functional.softmax(x_t, dim=-1)
            input = x_t
            time = torch.ones_like(t) # dummy t
        else:
            input, time = x_t, t

        predicted_x0_logits = self.model_predict(input.float(), time.float(), attn_mask).to(torch.float32) 
        loss = self.compute_loss(x, predicted_x0_logits, t, x_t_counts, zeta)

        if attn_mask is not None:
            loss = loss * attn_mask

 
        vb_loss = loss.mean() * self.t_max 
        vb_loss_log = loss.mean(dim = -1) * self.t_max

        if attn_mask is not None:
            vb_loss = vb_loss / attn_mask.mean() 
            vb_loss_log = vb_loss_log / attn_mask.mean() 
        print('loss is',vb_loss)

        # wandb log the loss for each batch torch.mean(loss, dim=-1), also log the loss for each t. 
        t_list = t.detach().cpu().tolist() if t.requires_grad else t.cpu().tolist()
        loss_list =vb_loss_log.detach().cpu().tolist() # loss is nxd. average over d
        for i in range(len(t_list)):
            wandb.log({"t_iter":float(t_list[i]), "loss_iter":float(loss_list[i])})

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

    def matrix_exponential_L_T(self, v):
        """
        v: shape (batch_size, 1). Rate.
        returns e^(v L^T)
        """ 
        scaled_eigenvals = v * self.eigenvalues  # broadcast over eigenvalues
        eigenval_exp = torch.diag_embed(torch.expm1(scaled_eigenvals))
        matrix_exponential = torch.transpose(self.eigenvectors_inv, 1, 0) @ eigenval_exp @ torch.transpose(self.eigenvectors, 1, 0)
        eye = torch.eye(matrix_exponential.shape[-1], device=matrix_exponential.device, dtype=torch.float32).unsqueeze(0)
        return matrix_exponential.double() + eye.double() ## will be real, because L and tau are real

    def get_posterior(self, x_t, pred_x0_logits, t, delta_t, zeta, corrector=False):
        """
        computes (pred(x_0)^T e^{tau_{t - eps} L}) o (x_t^T (e^{delta tau L^T}))
        Computes p(x_{t - eps} | x_t)
        """
        t_eps = t - delta_t
        delta_tau = self.tau(t) - self.tau(t-delta_t)
        delta_tau = self.tau(t).view(-1, 1, 1) - self.tau(t-delta_t).view(-1, 1, 1)
        if corrector:
            t_tensor = torch.ones_like(t_eps) * t
            first_term = self.compute_probabilities(pred_x0_logits.double(), t_tensor.double(), zeta).double() #  X^T e^(τ(t)L)
        else:
            first_term = self.compute_probabilities(pred_x0_logits.double(), t_eps.double(), zeta).double() #  X^T e^(τ(t)L)
        second_term = x_t.double() @ self.matrix_exponential_L_T(delta_tau).squeeze(1)
        return first_term * second_term
    
    def sample_step(self, x_t, t_tensor, delta_t_tensor, attn_mask, corrector=False):
        B, D, C = x_t.shape
        
        density = self.p_x_t_given_x_0(x_t, t_tensor, zeta=self.zeta)  # [n, D]
        
        # add SSP later
        if self.ssp:
            p_x_t_given_x_0 = density
            input = torch.log(self.p0).to(x_t.device) + p_x_t_given_x_0 # applying Bayes rule
            input[:,:,-1] -= 100 # padding token
            input = torch.nn.functional.softmax(input, dim=-1)
            time = torch.ones_like(t_tensor) # dummy t
        else:
            input, time = x_t, t_tensor
            
        pred_x0_logits = self.model_predict(input, time, attn_mask) + density
        
        posterior = self.get_posterior(x_t, pred_x0_logits, t_tensor, delta_t_tensor, self.zeta, corrector=corrector)  # [n, D, C]
        posterior = torch.clamp(posterior, min=1e-10)
        posterior /= posterior.sum(-1, keepdims=True)
        posterior = posterior.view(B * D, C) # [n*D, C]
        
        x_t_min_eps = torch.multinomial(posterior, num_samples=1)
        x_t_min_eps = F.one_hot(x_t_min_eps.squeeze(), num_classes=C).view(B, D, C).float()  # [n, D, C]
        
        return x_t_min_eps
    
    def sample_point(self, x, zeta, attn_mask=None, rand_shape=None):
        B, D = x.shape[:2]
        ts = torch.rand(x.shape[0], device=x.device).detach() * self.t_max
        
        with torch.no_grad():
            batch_size = x.shape[0]
            D = x.shape[1]
            
            probabilities = self.compute_probabilities(x, ts, zeta=zeta)
            probabilities = torch.clamp(probabilities, max=1.0) # adjust values that are over 1 due to numerical stability
            probabilities = torch.flatten(probabilities, start_dim=0, end_dim=1)
            
            S = np.full((batch_size * D,), zeta) # the counts for each D in each batch
            counts = torch.from_numpy(multinomial_numba(p=probabilities.detach().cpu().numpy(), S=S, num_classes=self.num_classes)).to(x.device)
            counts = counts.view(batch_size, D, self.num_classes)

            # rescale the counts Xt = √(1 − γt + γtζ)( Xt ζ − π)
            count_distribution = counts / zeta
            pi = self.get_stationary().to(x.device)
            x_t = torch.sqrt(1-self.gamma(ts) + self.gamma(ts) * zeta).view(-1,1,1) * (count_distribution - pi)
        
        return ts, x_t, counts
        

    def sample(self, batch_size, D, delta_t, device='cpu', max_D=None, corrector_steps=0, temp=1.0):
        max_D = D if max_D is None else max_D
        attn_mask = torch.ones((batch_size, max_D), dtype=torch.bool).to(device)
        attn_mask[:, D:] = 0
        C = self.num_classes
        pi = self.get_stationary()

        multi_input = pi.to(device).unsqueeze(0).expand(batch_size * max_D, -1)
        x_t = torch.multinomial(multi_input, num_samples=1).to(device)   # [n*D]
        x_t = F.one_hot(x_t.squeeze(), num_classes=C).view(batch_size, max_D, C).float()  # [n, D, C]
    
        t = self.t_max
        delta_t_tensor = torch.ones(size=(batch_size,), device=device) * delta_t

        with torch.no_grad():
            num_steps = int(t // delta_t)
            for _ in tqdm(range(num_steps), desc="Sampling", leave=False, disable=False):
                t_tensor = torch.ones(size=(batch_size,), device=device) * t
                
                x_t_min_eps = self.sample_step(x_t, t_tensor, delta_t_tensor, attn_mask, corrector=False)
                
                for step in range(corrector_steps - 1):
                    x_t_min_eps = self.sample_step(x_t_min_eps, t_tensor, delta_t_tensor, attn_mask, corrector=True)
                    
                t -= delta_t
                x_t = x_t_min_eps
            
            x_t = torch.argmax(x_t, dim=-1)
        
        return x_t[:, :D]

    def training_step(self, batch, batch_idx):
        x, attn_mask, extra = self.get_inputs(batch)

        # in training, randomly iterate between all zeta values if zeta is a list
        zeta = random.choice(self.zeta) if isinstance(self.zeta, list) else self.zeta 

        loss, info = self(x, attn_mask, zeta, extra=extra)
        if self.sample_x is None:
            self.sample_x = x[:1]
            self.sample_a = None if attn_mask is None else attn_mask[:1]
        
        perplexity = np.exp(info['ce_loss'])

        if batch_idx % 50 == 0:
            print(f"Loss is {loss.item()}")
            # print(f'Loss is {normalized_loss.item()}')
                
        self.log_dict({
            f'train_loss': info['vb_loss'],
            f'train_ce_loss': info['ce_loss'],
            f'train_perplexity': perplexity,
        }, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):        
        x, attn_mask, extra = self.get_inputs(batch)

        loss_dict = {}
        # get loss for each zeta 
        if isinstance(self.zeta, list):
            avg_vb_loss = 0
            avg_ce_loss = 0
            for z in self.zeta:
                loss, info = self(x, attn_mask, zeta=z, extra=extra)
                loss_dict[f'val_loss_{z}'] = info['vb_loss']
                loss_dict[f'val_ce_loss_{z}'] = info['ce_loss']
                avg_vb_loss += info['vb_loss']
                avg_ce_loss += info['ce_loss']
            avg_vb_loss = avg_vb_loss/len(self.zeta)
            avg_ce_loss = avg_ce_loss/len(self.zeta)
            perplexity = np.exp(avg_ce_loss)
            # normalized_loss = np.exp(avg_ce_loss / (x.shape[0] * x.shape[1]))

            loss_dict["val_loss"] = avg_vb_loss
            loss_dict["val_ce_loss"] = avg_ce_loss
            loss_dict["val_perplexity"] = perplexity
            # loss_dict["val_normalized_loss"] = normalized_loss
            # TODO - add normalizes loss to validation step

        else: # just one zeta. Behaves normally
            loss, info = self(x, attn_mask, zeta=self.zeta, extra=extra)
            perplexity = np.exp(info['ce_loss'])

            np_loss = loss.detach().cpu().numpy()
            # normalized_loss = np.exp(np_loss/ (x.shape[0] * x.shape[1]))
            loss_dict['val_loss'] = info['vb_loss']
            loss_dict['val_ce_loss'] = info['ce_loss']
            loss_dict['val_perplexity'] = perplexity
            
        self.log_dict(loss_dict, on_step=False, on_epoch=True, sync_dist=True)
        return loss_dict


class DiscreteDiffusion(WFDiffusion):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.zeta = 1

    def forward(self, x, attn_mask, **kwargs):
        if "zeta" in kwargs:
            kwargs.pop("zeta")
        return super().forward(x, attn_mask, zeta=1, **kwargs)
