import math
import torch
import torch.nn.functional as F

from .trainer import ContinuousTimeDiffusion
from .utils.utils import convert_to_probs


class DiscreteDiffusionLanguage(ContinuousTimeDiffusion):
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes,
        forward_kwargs={"type":"uniform", 
                        "ssp":"false"},
        schedule_type="cos",
        logistic_pars=False,
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

        self.ssp = forward_kwargs['ssp']



    def calculate_tau(self):
        eigval_1 = 1
        tau = lambda t, zeta: 1/eigval_1 * torch.log(math.sqrt(zeta) * torch.sqrt((self.gamma(t)+1/zeta) / (1 - self.gamma(t))))
        tau_prime = lambda t, zeta:  (zeta + 1) / (eigval_1 * 2 * (zeta * self.gamma(t) + 1)*(1-self.gamma(t))) * self.gamma_prime(t)
     
        return tau, tau_prime

        
    def pre_configure_model(self, dataloader):
        """
        Assumes linear schedule_condition. No other schedule_condition is currently implemented
        """
        self.calc_p0(dataloader)
        self.gamma = lambda t: t 
        self.gamma_prime = lambda t: 1

    
    def get_stationary(self):
        # TODO - take in HP - stationary distribution
        stationary = torch.ones(self.num_classes)
        stationary = stationary / self.num_classes
        return stationary
    

    def sample_point(self, x, zeta, attn_mask=None, rand_shape=None):   
        with torch.no_grad():
            # breakpoint()
            batch_size = x.shape[0]
            D = x.shape[1]
            ts = torch.rand(x.shape[0], device=x.device).detach() * self.t_max # t is uniformly distributed. Each dimension d has the same t 
            
            probabilities = self.compute_probabilities(x, ts, zeta=zeta) # [B, L, C]
            probabilities = torch.clamp(probabilities, max=1.0) # adjust values that are over 1 due to numerical stability
            probabilities = torch.flatten(probabilities, start_dim=0, end_dim=1) # [B * L, C]
            
            counts = torch.rand(size=(batch_size * D, 1), device=x.device) # [B * L, 1]
            cum_probs = torch.cumsum(probabilities, dim=-1)
            counts = torch.searchsorted(cum_probs, counts)
            counts = torch.clamp(counts, min=0, max=self.num_classes - 1)

            counts = F.one_hot(counts, num_classes=self.num_classes)

            counts = counts.view(batch_size, D, self.num_classes)

            # rescale the counts Xt = √(1 − γt + γtζ)( Xt ζ − π)
            count_distribution = counts / zeta
            pi = self.get_stationary().to(x.device)
            x_t = torch.sqrt(1-self.gamma(ts) + self.gamma(ts) * zeta).view(-1,1,1) * (count_distribution - pi)
            return ts, x_t, counts


    @torch.compile
    def right_multiply_with_matrix_exponential(self, v, tau_t):
        e_neg_tau_t = torch.exp(-1 * tau_t)
        v_pi = (v.float() @ self.pi).unsqueeze(-1)

        return e_neg_tau_t * v + (1 - e_neg_tau_t) * v_pi


    @torch.compile
    def left_multiply_with_matrix_exponential(self, v, tau_t):
        e_neg_tau_t = torch.exp(-1 * tau_t).unsqueeze(-1)

        return e_neg_tau_t * v + (1 - e_neg_tau_t) * self.pi * (v.sum(-1).unsqueeze(-1))


    def compute_probabilities(self, x, t, zeta):
        '''
        Returns array of  X^T e^(τ(t)L) with shape (batch_size x D x B)

        zeta will be ignored - equal to 1
        '''
        x = convert_to_probs(x, self.num_classes).float()
        tau_t = self.tau(t, 1).view(-1, 1)
        probabilities = self.left_multiply_with_matrix_exponential(x, tau_t)
        return probabilities
   

    def compute_loss(self, x, predicted_x0_logits, t, x_t_counts):
        p = self.compute_probabilities(x, t, zeta=1)
        q = self.compute_probabilities(predicted_x0_logits, t, zeta=1)

        p = p / (p * x_t_counts).sum(-1)[..., None]
        q = q / (q * x_t_counts).sum(-1)[..., None]
        kl = p * torch.log(p/q) + q - p

        L = (self.pi[None, None, :] * x_t_counts).sum(-1)[..., None]

        out = (L * kl).sum(-1).sum(-1)

        betas = (self.beta(t, zeta=1) * out).sum(-1)
        return betas


    def p_x_t_given_x_0(self, x_t_counts, ts):
        """
        x_t_counts is a tensor of the counts of each letter. Shape n x D x B
        computes log p(x_t^d|x_0^d,s_t^d)
        by computing sum_{alphabet} X_t * log( X_0^T e^(τ(t)L))
        """
        tau_t = self.tau(ts, 1).view(-1, 1, 1)
        probabilities = self.right_multiply_with_matrix_exponential(x_t_counts, tau_t)
        return probabilities 
    
    def forward(self, x, attn_mask=None, zeta=None):
        """
        x: sequences described by letter indices. shape: batch_size x max(L)
        """
        t, x_t, x_t_counts = self.sample_point(x, zeta, attn_mask)
        p_x_t_given_x_0 = self.p_x_t_given_x_0(x_t_counts=x_t_counts, ts=t)

        if self.ssp:
            x_t = torch.log(self.p0).to(x.device) + p_x_t_given_x_0 # applying Bayes rule
            x_t[:,:,-1] -= 100 # padding token
            x_t = torch.nn.functional.softmax(x_t, dim=-1)
        
        NN_predicted_x0_logits = self.model_predict(x_t, t, attn_mask).to(torch.float32) 
        predicted_x0_logits = NN_predicted_x0_logits + p_x_t_given_x_0

        loss = self.compute_loss(x, predicted_x0_logits, t, x_t_counts)
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
        
        return vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }