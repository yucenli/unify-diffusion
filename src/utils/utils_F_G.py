import torch
import scipy
import logging
import mpmath as mp
from tqdm import tqdm


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mp_mul(a, b):
    # element-wise multiplication of two mpmath matrices
    assert a.rows == b.rows and a.cols == b.cols
    result = mp.matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            result[i,j] = a[i,j] * b[i,j]
    return result


def mp_element_wise(mp_func, x):
    return mp.matrix([mp_func(xi) for xi in x])


def mp_alternating_sum(terms):
    total = mp.mpf('0')
    for i, term in enumerate(terms):
        if i % 2 == 0:
            total += term
        else:
            total -= term
    return total


def mp_hyp2f1(a_vec, b_vec, c, z):
    # compute hyp2f1 for vectors a_vec, b_vec with same length
    assert a_vec.rows == b_vec.rows and a_vec.cols == 1 and b_vec.cols == 1
    result = mp.matrix(a_vec.rows, 1)
    for i in range(a_vec.rows):
        result[i] = mp.hyp2f1(a_vec[i], b_vec[i], c, z)
    return result


def compute_series_precise_individual(t, C, x_t_val, max_k, series='G'):
    psi = mp.mpf(C)
    
    k = mp.matrix(mp.arange(max_k))
    if series == 'G':
        log_exp_term = mp_mul(-k, (k + psi - 1)) * t / 2
        log_poch = mp_element_wise(mp.loggamma, psi + k - 1) - mp.loggamma(psi)
        log_fact = mp_element_wise(mp.loggamma, k + 1)
        factorial_term = mp_element_wise(mp.exp, log_poch - log_fact + log_exp_term)
        coeff_term = mp_mul(factorial_term, 2 * k + psi - 1)
        
        hyp_terms = mp_hyp2f1(-k, psi + k - 1, psi * (1/C), x_t_val)
    else:
        log_exp_term = mp_mul(-k, (k + psi + 1)) * t / 2
        log_poch = mp_element_wise(mp.loggamma, psi + k) - mp.loggamma(psi)
        log_fact = mp_element_wise(mp.loggamma, k + 1)
        factorial_term = mp_element_wise(mp.exp, log_poch - log_fact + log_exp_term)
        c_term = mp_mul(2 * k + psi + 1, psi + k) / ((psi + 1) * psi)
        coeff_term = mp_mul(factorial_term, c_term)
        
        hyp_terms = mp_hyp2f1(-k, psi + k + 1, psi * (1/C) + 1, x_t_val)
        
    terms = mp_mul(coeff_term, hyp_terms)
    series_sum = mp_alternating_sum(terms)
    return float(series_sum)


@torch.compile(mode="max-autotune-no-cudagraphs")
def torch_hyp2f1(psi, max_k, x, series='G'):
    batch_shape = x.shape
    x_expand = x.unsqueeze(-1)

    M = torch.zeros(*batch_shape, max_k, dtype=torch.float64, device=x.device)
    j = torch.arange(max_k, dtype=torch.float64, device=x.device)

    M_prev_2 = torch.ones_like(M)
    # psi + j + 1 for F, psi + j - 1 for G
    if series == 'F':
        psi_j_term = psi + j + 1
        c = 2.0
    else:
        psi_j_term = psi + j - 1
        c = 1.0        
    M_prev_1 = torch.ones_like(M) - psi_j_term * x_expand / c
    
    M[..., 0] = M_prev_2[..., 0]
    M[..., 1] = M_prev_1[..., 1]
    
    for i in range(0, max_k-2):
        denom = c + i + 1
        a = (c + 2 * i + 2 - (i + psi_j_term + 1) * x_expand)
        b = (i + 1) * (1 - x_expand)
        M_curr = (a / denom) * M_prev_1 - (b / denom) * M_prev_2
        M_prev_2 = M_prev_1
        M_prev_1 = M_curr
        
        M[..., i + 2] = M_curr[..., i + 2]
    
    return M


def a_ks_psi_torch(ks, psi, t, pi_x0, x_t_x0):
    # t: B, ks: len(ks), log_exp_term: B x 1 x 1 x len(ks)
    log_exp_term = (-ks * (ks + psi - 1) * t.unsqueeze(-1) / 2).reshape(-1, 1, 1, len(ks))
    
    # len(ks)
    log_poch = torch.special.gammaln(psi + ks - 1) - scipy.special.gammaln(psi)
    log_fact = torch.special.gammaln(ks + 1)
    
    coeff_term = (2 * ks + psi - 1) * torch.exp(log_poch - log_fact + log_exp_term)

    # B x D x C x len(ks)
    hyp_term = torch_hyp2f1(psi, len(ks), x_t_x0, series='G')
    return coeff_term * hyp_term


def b_ks_psi_torch(ks, psi, t, pi_x0, x_t_x0): 
    log_exp_term = (-ks * (ks + psi + 1) * t.unsqueeze(-1) / 2).reshape(-1, 1, 1, len(ks))
          
    log_poch = torch.special.gammaln(psi + ks) - scipy.special.gammaln(psi)
    log_fact = torch.special.gammaln(ks + 1)
    factorial_term = torch.exp(log_poch - log_fact + log_exp_term)
    
    coeff_term = factorial_term * (2 * ks + psi + 1) * (psi + ks) / ((psi + 1) * psi)
        
    # hyp_term = scipy.special.hyp2f1(-ks, psi + ks + 1, psi * pi_x0.unsqueeze(-1) + 1, x_t_x0.unsqueeze(-1))
    hyp_term = torch_hyp2f1(psi, len(ks), x_t_x0, series='F')
    return coeff_term * hyp_term


def compute_series_torch(series, t, x_0, x_t, max_terms):
    device = x_t.device
    t = t.double()
    x_t = x_t.double()
    
    if series == 'G':
        term_func = a_ks_psi_torch
    elif series == 'F':
        term_func = b_ks_psi_torch
    else:
        raise ValueError("series must be 'G' or 'F'")

    C = x_t.shape[-1]
    pi_x0 = torch.ones_like(x_t) / C # assuming uniform prior
    x_t_x0 = x_t
        
    ks = torch.arange(max_terms).to(device).double()
    
    terms = term_func(ks, C, t, pi_x0, x_t_x0) # B x D x C x max_terms
    checkerboard = torch.tensor([1 if i % 2 == 0 else -1 for i in range(max_terms)], dtype=torch.float64)
    result_terms = terms * checkerboard.to(terms.device)
    result = result_terms.sum(dim=-1)
    abs_result = terms.abs().sum(dim=-1)
    
    bad_condition = (torch.abs(result / abs_result) < 1e-11)
    
    return result, bad_condition


def compute_series_precise_bad(series, t, x_0, x_t, max_terms, retry):
    t_cpu = t.cpu()
    result = torch.ones_like(x_t).double()
    x_t_x0_cpu = x_t.cpu()
    C = x_t.shape[-1]

    retry_indices = retry.nonzero(as_tuple=False)
    for idx in tqdm(retry_indices, desc="Recomputing bad conditions", total=len(retry_indices)):
        b, d, c = idx
        result[b, d, c] = compute_series_precise_individual(t_cpu[b].item(), C, x_t_x0_cpu[b, d, c].item(), max_terms, series)
            
    return result
            