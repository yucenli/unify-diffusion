import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from transformers import BertModel
import faiss
import scipy
from scipy.sparse import coo_matrix
import numba
import logging
import mpmath as mp
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _at(a, t, x):
    # t is 1-d, x is integer value of 0 to num_classes - 1
    bs = t.shape[0]
    t = t.reshape((bs, *[1] * (x.dim() - 1)))
    return a[t, x, :]

def kls(dist1, dist2, eps=None): # KL of dists on last dim
    assert not torch.isnan(torch.log_softmax(dist2, dim=-1)).any(), f"torch.log_softmax(dist2, dim=-1 ) {torch.log_softmax(dist2, dim=-1)}"
    assert not torch.isnan(torch.log_softmax(dist1, dim=-1)).any(), f"torch.log_softmax(dist1, dim=-1), {torch.log_softmax(dist1, dim=-1)}"
    out = F.kl_div(torch.log_softmax(dist2, dim=-1),
                   torch.log_softmax(dist1, dim=-1),
                  log_target=True, reduction='none').sum(-1)
    return out



def convert_to_distribution(x_0, num_classes, eps):
    # returns log probs of x_0 as a distribution
    if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
        x_0_logits = torch.log(
            torch.nn.functional.one_hot(x_0, num_classes) + eps
        )
    else:
        x_0_logits = x_0.clone()
    return x_0_logits

def convert_to_probs(x_0, num_classes):
    # returns probs of x_0 as a distribution. input is either indices or logits
    if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
        x_0_probs = torch.nn.functional.one_hot(x_0, num_classes)
        assert torch.sum(torch.sum(x_0_probs, dim=-1) == 0) == 0, f"torch.sum(torch.sum(x_0_probs, dim=-1)), {torch.sum(torch.sum(x_0_probs, dim=-1))}"
    else:
        x_0_probs = torch.softmax(x_0.clone(), dim=-1)
    return x_0_probs

def get_inf_gen(forward_kwargs, num_classes):
    if forward_kwargs['type'] == "uniform":
        L = torch.ones(num_classes, num_classes) / (num_classes-1)
        L.diagonal().fill_(-1)
    elif forward_kwargs['type'] == "gaussian":
        bandwidth = forward_kwargs['bandwidth']
        range_ = torch.arange(num_classes)
        diff_mat = (range_[:, None] - range_[None, :]) ** 2
        L = torch.exp(- diff_mat / (2 * (bandwidth * num_classes) ** 2))
        L = L / (L.sum(-1).max() - 1)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    elif forward_kwargs['type'] == "blosum":
        from evodiff.utils import Tokenizer
        tokenizer = Tokenizer()
        # from https://web.expasy.org/protscale/pscale/A.A.Swiss-Prot.html
        aa_freq = np.array([8.25, 5.53, 4.06, 5.45, 1.37, 3.93, 6.75,
                            7.07, 2.27, 5.96, 9.66, 5.84, 2.42, 3.86,
                            4.70, 6.56, 5.34, 1.08, 2.92, 6.87] + 11*[0]) / 100 
        blosum_alphabet = np.array(list('ARNDCQEGHILKMFPSTWYVBZXJOU-'))
        tok_alphabet = np.array(tokenizer.alphabet)
        with open('data/blosum62-special-MSA.mat') as f:
            load_matrix = np.array([line.split()[1:] for line in f if line[0] in blosum_alphabet], dtype=int)
        map_ = blosum_alphabet[:, None] == tok_alphabet[None, :]
        blosum_matrix = np.zeros((len(tok_alphabet), len(tok_alphabet)))
        for i, ind_i in enumerate(np.argmax(map_, axis=1)):
            for j, ind_j in enumerate(np.argmax(map_, axis=1)):
                blosum_matrix[ind_i, ind_j] = load_matrix[i, j]
        # X_ij = BLOSUM_ij * p(aa_j) = p(aa_j | aa_i)
        cond_liks = (2. ** (blosum_matrix/2)) * aa_freq[None, :] 
        cond_liks = cond_liks ** forward_kwargs['beta']
        cond_liks = cond_liks / cond_liks.sum(-1)[:, None]
        L = cond_liks - np.eye(len(cond_liks))
        # break up
        l, V = np.linalg.eig(cond_liks[:20, :20])
        V_inv = np.linalg.inv(V)

        # alpha
        alpha = forward_kwargs['alpha']
        if alpha > 0:
            evals = (l**alpha - 1)[None, :] / alpha
        else:
            evals = np.log(l)
        L[:20, :20] = (V * evals) @ V_inv
        L[20:] *= - np.diagonal(L).min()
        L[L<0] = 0
        L = torch.tensor(L).float()
        range_ = torch.arange(num_classes)
        L[range_, range_] = -L.sum(-1)
    if ("make_sym" in forward_kwargs.keys() and forward_kwargs['make_sym']):
        L = (L + L.T) / 2
        range_ = torch.arange(num_classes)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    if (("normalize" in forward_kwargs.keys() and forward_kwargs['normalize'])
        or ("normalized" in forward_kwargs.keys() and forward_kwargs['normalized'])):
        L = L / (- L.diagonal()[:, None])
        range_ = torch.arange(num_classes)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    return L

def get_sort_S(S):
    S_flat, sort = torch.sort(S.flatten(), descending=True)
    S_sort = S_flat.reshape(S.shape)
    unsort = torch.zeros_like(sort)
    unsort[sort] = torch.arange(len(S_flat), device=S_flat.device)
    return S_sort, sort, unsort
    
def get_counts_S_flat(S_flat):
    unique, counts = torch.unique(torch.clamp(S_flat, min=0), return_counts=True)
    full_counts = torch.zeros(unique.max()+1, device=unique.device, dtype=torch.long)
    full_counts[unique] = counts
    return full_counts.flip(0).cumsum(0)

def _pad(tokenized, value, dim=2):
    """
    Utility function that pads batches to the same length.

    tokenized: list of tokenized sequences
    value: pad index
    """
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    if dim == 3: # dim = 3 (one hot)
        categories = tokenized[0].shape[-1]
        output = torch.zeros((batch_size, max_len, categories)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t), :] = t
    elif dim == 2: # dim = 2 (tokenized)
        output = torch.zeros((batch_size, max_len)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t)] = t
    else:
        print("padding not supported for dim > 3")
    return output

def sample_index_S(S):
    # Flatten the array
    S_flat = S.flatten()
    
    # Ensure all values are non-negative
    if torch.any(S_flat < 0):
        raise ValueError("All entries in S must be non-negative for probability sampling")
    
    # Sample an index
    sampled_flat_index = torch.multinomial(S_flat, num_samples=1)
    
    # Convert the flat index back to multidimensional index
    sampled_index = np.unravel_index(sampled_flat_index.item(), S.shape)
    
    return sampled_index

def log1p(x):
    result = torch.log1p(x)
    mask = x < -0.7
    if mask.any():
        x_neg = x[mask]
        neg_x = -x_neg
        inv_xp1 = 1.0 / (neg_x - 1.0)
        result[mask] = torch.log1p(inv_xp1) + torch.log(neg_x)
    return result


@numba.jit
def multinomial_numba(p, S, num_classes):
    """
    p is a numpy array of probabilities. array of shape B or shape len(S) x B
    S is a single dimensional numpy array of the total number of counts 
    """
    batch_size = len(S)
    result = np.zeros((batch_size, num_classes), dtype=np.int64)
    
    for i in range(batch_size):
        s_remaining = S[i]
        p_remaining = 1.0
        
        for j in range(num_classes - 1):
            if s_remaining <= 0:
                break
            p_j = p[i, j] if p.ndim == 2 else p[j]
            p_ratio = p_j / p_remaining 
            if p_ratio > 1: # clamp numbers slighly above 1 due to numerical stability
                p_ratio = 1.0
            if p_ratio < 0: 
                p_ratio=0.0
            count = np.random.binomial(s_remaining, p_ratio)
            result[i, j] = count
            s_remaining -= count
            p_remaining -= p_j
        
        result[i, -1] = s_remaining
    
    return result

def evodiff_to_esm_tokenization(input):
    # Expects tensor containing evodiff tokenizer indices
    evodiff_to_esm_indices = torch.tensor([ 5, 23, 13,  9, 18,  6, 21, 12, 15,  4, 20, 17, 14, 16, 10,  8, 11,  7,
        22, 19, 25, 27, 24,  1, 28, 26,  1,  2, 32,  0,  1]).to(input.device)
    return evodiff_to_esm_indices[input]


def compute_coefficient_precise(m, k, psi, tau):
    if k < m:
        return mp.mpf(0)
    elif k == 0 and m == 0:
        return mp.mpf(1)
    
    log_num = (
        mp.log(2*k + psi - 1) +
        mp.loggamma(psi + m + k - 1) - mp.loggamma(psi + m)
    )
    log_den = mp.loggamma(m + 1) + mp.loggamma(k - m + 1)
    log_exp = -k * (k + psi - 1) * tau / 2

    log_c = log_num - log_den + log_exp
    return mp.exp(log_c)


def mp_sub(a, b):
    assert len(a) == len(b)
    return [a_i - b_i for a_i, b_i in zip(a, b)]


def mp_simulate_ancestral_process(tau, psi, U):
    mp.mp.dps = 50
    k_sums = []
    prev_terms = []
    num_terms = []
    
    upper = mp.mpf(0)
    lower = mp.mpf(0)
    for M in range(10000):
        assert M == len(k_sums)         
            
        k_sums.append(mp.mpf(0))
        prev_terms.append(mp.mpf(0))
        num_terms.append(M)
        
        # find k_start for M
        current_term = mp.mpf(0)
        while True:
            k_M = num_terms[M]
            current_term = compute_coefficient_precise(M, k_M, psi, tau)
            k_sums[M] += ((-1) ** (k_M - M)) * current_term
            num_terms[M] += 1   # number of terms used in k_sums
            
            if (prev_terms[M] >= current_term):
                # make sure upper >= lower
                if (num_terms[M] % 2 != (M % 2)):
                    break
                
            prev_terms[M] = current_term
        
        prev_terms[M] = current_term  

        upper += k_sums[M]
        lower += k_sums[M] - prev_terms[M]
        assert k_sums[M] > (k_sums[M] - prev_terms[M])

        while (lower < U) and (U < upper):
            for m in range(M+1):
                c1 = compute_coefficient_precise(m, num_terms[m], psi, tau)
                c2 = compute_coefficient_precise(m, num_terms[m] + 1, psi, tau)
                assert prev_terms[m] - c1 > 0 # lower bound increases
                assert -c1 + c2 < 0 # upper bound decreases

                lower += prev_terms[m] - c1
                upper += -c1 + c2
                prev_terms[m] = c2
                
                num_terms[m] += 2
            
        if lower >= U:
            return M
        elif upper <= U:
            continue
        else:
            raise ValueError("U must be between lower and upper")
    
    raise RuntimeError("Should not reach here")   


def compute_coefficients(m, k, psi, tau):
    # broadcast k, m
    m, k, tau = torch.broadcast_tensors(m, k, tau)
    # assert k.shape == m.shape, "k and m must have the same shape"
    mask = (k >= m)

    # log terms
    log_num = (
        torch.log(2*k + psi - 1) +
        torch.lgamma(psi + m + k - 1) - torch.lgamma(psi + m)
    )
    log_den = torch.lgamma(m + 1) + torch.lgamma(k - m + 1)
    log_exp = -k * (k + psi - 1) * tau / 2

    log_c = log_num - log_den + log_exp

    log_c = log_c.masked_fill(torch.isnan(log_c), float('-inf'))
    # set k < m to -inf
    log_c = log_c.masked_fill(~mask, float('-inf'))

    return torch.exp(log_c)


def increase_ks(coeffs, psi, tau):
    m_size, k_size = coeffs.shape[-2:]
    new_ks = torch.arange(k_size, k_size * 2).unsqueeze(0).to(coeffs)
    new_ms = torch.arange(m_size).unsqueeze(1).to(coeffs)
    new_coeffs = compute_coefficients(new_ms, new_ks, psi, tau)
    coeffs = torch.cat([coeffs, new_coeffs], dim=-1)
    logger.debug(f"K New coeffs shape: {new_coeffs.shape} {coeffs.shape}")
    return coeffs


def increase_ms(coeffs, psi, tau):
    m_size, k_size = coeffs.shape[-2:]
    new_ks = torch.arange(k_size).unsqueeze(0).to(coeffs)
    new_ms = torch.arange(m_size, m_size + 32).unsqueeze(1).to(coeffs)
    new_coeffs = compute_coefficients(new_ms, new_ks, psi, tau)
    coeffs = torch.cat([coeffs, new_coeffs], dim=-2)
    logger.debug(f"M New coeffs shape: {new_coeffs.shape} {coeffs.shape}")
    return coeffs


def simulate_ancestral_process_inner(tau, U, psi, coeffs, max_m=64, max_k=1024):
    while coeffs.shape[-2] <= max_m and coeffs.shape[-1] <= max_k:
        # make sure ks start decreasing
        while coeffs.shape[-1] <= max_k:
            if (coeffs[..., -2] <= coeffs[..., -3]).all():
                break
            coeffs = increase_ks(coeffs, psi, tau)
            
        rows = torch.arange(coeffs.shape[-2]).unsqueeze(1).to(tau)
        cols = torch.arange(coeffs.shape[-1]).unsqueeze(0).to(tau)
        checkerboard = ((rows + cols) % 2 == 0) * 2 - 1
        alternating_coeffs = coeffs * checkerboard
    
        S = alternating_coeffs.cumsum(-1) # sum over ks
        # alternate between -1 and -2 for "rows + 1"
        zigzag = ((torch.arange(coeffs.shape[-2] + 1) % 2 == 0).to(tau) * 1 - 2).long()
        b_idx = torch.arange(S.size(0)).unsqueeze(1)
        m_idx = torch.arange(S.size(1)).unsqueeze(0)
        lower, upper = S[b_idx, m_idx, zigzag[:-1]], S[b_idx, m_idx, zigzag[1:]]
        
        assert (lower <= upper).all()
        S_lower = lower.cumsum(-1) # sum over ms
        
        filter = S_lower > U
        batch_done = filter.any(dim=1)
        if batch_done.all():
            m = torch.argmax(filter.int(), dim=1)
            return m, batch_done, coeffs
        else:
            S_upper = upper.sum(-1)
            more_precision_needed = (U < S_upper).any(1)
            # if any batch needs more precision, increase ks
            if more_precision_needed.any():
                coeffs = increase_ks(coeffs, psi, tau)
            
            # if any batch doesn't need more precision, increase ms
            if (~more_precision_needed).any():
                logger.debug("M increase needed")
                coeffs = increase_ms(coeffs, psi, tau)
    
    m = torch.argmax(filter.int(), dim=1)
    return m, batch_done, coeffs


def simulate_ancestral_process_torch(tau, psi, U):
    """
    Assumes taus is shape B x D, U is shape B x D
    """
    assert (tau >= 0.05).all(), "tau must be greater than 0.05"
    batch_shape = tau.shape
    tau = tau.reshape(-1, 1, 1).double() # shape (BxD, 1, 1)
    U = U.reshape(-1, 1).double() # shape (BxD, 1)
    
    start_m, start_k = 32, 64
    max_m, max_k = 64, 128
    ms = torch.arange(start_m).unsqueeze(1).to(tau)
    ks = torch.arange(start_k).unsqueeze(0).to(tau)
    coeffs = compute_coefficients(ms, ks, psi, tau)
    coeffs[..., 0, 0] = 1.0 # k = m = 0
    
    condition = torch.zeros(len(U), dtype=torch.float64).to(tau.device)
    
    m = torch.ones(len(U), dtype=torch.long).to(tau.device) * -1
    batch_done = torch.zeros(len(U), dtype=torch.bool).to(tau.device)
    
    for i in range(5):
        incomplete = ~batch_done
        
        # coeffs has shape (B_incomplete, m, k)
        assert coeffs.shape[0] == tau[incomplete].shape[0]
        m_sub, batch_sub, coeffs_sub = simulate_ancestral_process_inner(
            tau[incomplete], U[incomplete], psi, coeffs, max_m=max_m, max_k=max_k)
        m[incomplete] = m_sub
        batch_done[incomplete] = batch_sub
        
        condition[incomplete] = (coeffs_sub.sum(dim=(-2, -1)) / (coeffs_sub.abs().sum(dim=(-2, -1)))).squeeze()
                
        if batch_done.all():
            break
        else:
            logger.debug(f"{(~batch_done).sum().item()} batches failed to converge in m={coeffs_sub.shape[-2]}, k={coeffs_sub.shape[-1]}")
            max_m *= 2
            max_k *= 2
            coeffs = coeffs_sub[~batch_sub]
    
    recompute = (condition < 1e-11) | (~batch_done)

    return m.reshape(batch_shape), recompute.reshape(batch_shape)


def simulate_ancestral_process_mixed(tau, psi, U=None):
    """
    Assumes tau is shape B x D, but all Ds are the same
    """
    assert (tau >= 0.05).all(), "tau must be greater than 0.05"
    assert (tau.min(1)[0] == tau.max(1)[0]).all(), "all taus in batch must be the same"
    
    batch_shape = tau.shape
    results = torch.zeros_like(tau, dtype=torch.long)
    
    if U is None:
        U = torch.rand_like(tau)

    # for small tau, use mp
    small_mask = tau[:, 0] < 0.07
    small_mask_idx = (small_mask).nonzero(as_tuple=True)[0]
    for i in small_mask_idx:
        ms = mp_simulate_ancestral_process_batch(tau[i, 0].item(), float(psi), U[i])
        results[i] = torch.tensor(ms, dtype=torch.long).to(tau.device)
    
    # for large tau, use torch
    m_large, recompute_large = simulate_ancestral_process_torch(tau[~small_mask], psi, U[~small_mask])
    results[~small_mask] = m_large.to(tau.device)

    # for those that need recompute, use mp
    recompute_indices = (recompute_large).nonzero(as_tuple=False)
    if recompute_large.sum() > 0:
        logger.warning(f"Recomputing {len(recompute_indices)} samples with mp")
    for i, j in tqdm(recompute_indices, desc="Recomputing with mp", disable=True):
        m = mp_simulate_ancestral_process(tau[i, j].item(), float(psi), U[i, j].item())
        results[i, j] = m
    m = results

    assert (not (m < 0).any()), "m should be non-negative"
    return m.reshape(batch_shape)



def mp_compute_coefficients(ms, ks, psi, tau):
    precise_ms = [mp.mpf(float(m)) for m in ms]
    precise_ks = [mp.mpf(float(k)) for k in ks]
    
    result = mp.matrix(len(precise_ms), len(precise_ks))
    for i in range(len(precise_ms)):
        for j in range(len(precise_ks)):
            result[i, j] = compute_coefficient_precise(precise_ms[i], precise_ks[j], psi, tau)
    
    return result


def mp_compute_alternating_cumsums(coeffs):
    # checkerboard cum sum over each row
    rows, cols = coeffs.rows, coeffs.cols
    result = mp.matrix(rows, cols)
    lowers = [mp.mpf('0')] * rows
    uppers = [mp.mpf('0')] * rows
    
    for i in range(rows):
        checkerboard = [1 if (i+j) % 2 == 0 else -1 for j in range(cols)]
        result[i, 0] = coeffs[i, 0] * checkerboard[0]
        for j in range(1, cols):
            result[i, j] = result[i, j-1] + coeffs[i, j] * checkerboard[j]
            
        if result[i, cols-2] < result[i, cols-1]:
            lowers[i] = result[i, cols-2]
            uppers[i] = result[i, cols-1]
        else:
            lowers[i] = result[i, cols-1]
            uppers[i] = result[i, cols-2]
            
    return result, lowers, uppers


def mp_matrix_concat(a, b, axis=0):
    if axis == 0:
        if a.cols != b.cols:
            raise ValueError("Matrices must have the same number of columns to concatenate along rows.")
        result = mp.matrix(a.rows + b.rows, a.cols)
        for i in range(a.rows):
            for j in range(a.cols):
                result[i, j] = a[i, j]
        for i in range(b.rows):
            for j in range(b.cols):
                result[a.rows + i, j] = b[i, j]
        return result
    elif axis == 1:
        if a.rows != b.rows:
            raise ValueError("Matrices must have the same number of rows to concatenate along columns.")
        result = mp.matrix(a.rows, a.cols + b.cols)
        for i in range(a.rows):
            for j in range(a.cols):
                result[i, j] = a[i, j]
        for i in range(b.rows):
            for j in range(b.cols):
                result[i, a.cols + j] = b[i, j]
        return result


def mp_increase_ks(coeffs, cumsums, lowers, uppers, psi, tau):
    m_size, k_size = coeffs.rows, coeffs.cols
    new_ks = np.arange(k_size, k_size * 2)
    new_ms = np.arange(m_size)
    new_coeffs = mp_compute_coefficients(new_ms, new_ks, psi, tau)
        
    # update first column of new_cumsums with last column of old coeffs
    new_cumsums = new_coeffs.copy()
    for i in range(m_size):
        sign = 1 if (i + k_size) % 2 == 0 else -1
        new_cumsums[i, 0] += sign * cumsums[i, k_size-1]
    new_cumsums, lowers, uppers = mp_compute_alternating_cumsums(new_cumsums)
    
    combined_coeffs = mp_matrix_concat(coeffs, new_coeffs, axis=1)
    combined_cumsums = mp_matrix_concat(cumsums, new_cumsums, axis=1)
    
    return combined_coeffs, combined_cumsums, lowers, uppers


def mp_increase_ms(coeffs, cumsums, lowers, uppers, psi, tau):
    m_size, k_size = coeffs.rows, coeffs.cols
    new_ms = np.arange(m_size, m_size * 2)
    new_ks = np.arange(k_size)
    new_coeffs = mp_compute_coefficients(new_ms, new_ks, psi, tau)
    
    new_cumsums, new_lowers, new_uppers = mp_compute_alternating_cumsums(new_coeffs)
    
    combined_coeffs = mp_matrix_concat(coeffs, new_coeffs, axis=0)
    combined_cumsums = mp_matrix_concat(cumsums, new_cumsums, axis=0)
    combined_lowers = lowers + new_lowers
    combined_uppers = uppers + new_uppers
    
    return combined_coeffs, combined_cumsums, combined_lowers, combined_uppers


def mp_find_smallest_ms_greater_than_U(lowers, Us):
    complete = True
    
    results = [-1] * len(Us)
    
    for idx, U in enumerate(Us):
        cum_lower = mp.mpf(0)
        for i in range(len(lowers)):
            cum_lower += lowers[i]
        
            if cum_lower >= U:
                results[idx] = i
                break
        complete = complete and (results[idx] >= 0)
        
    return results, complete


def mp_more_precision_needed(uppers, Us):
    cum_upper = mp.fsum(uppers)
    for U in Us:
        if U < cum_upper:
            return True
    return False


def mp_is_decreasing(coeffs):
    n_cols = coeffs.cols
    for i in range(coeffs.rows):
        if coeffs[i, n_cols-2] > coeffs[i, n_cols-3]:
            return False
    return True


def mp_simulate_ancestral_process_batch(tau, psi, Us):
    mp.mp.dps = 50  # set decimal places for high precision
    
    tau = mp.mpf(tau)
    psi = mp.mpf(psi)
    Us = [mp.mpf(float(U)) for U in Us]
    
    start_m, start_k = 2, 4
    ms = np.arange(start_m)
    ks = np.arange(start_k)
    coeffs = mp_compute_coefficients(ms, ks, psi, tau)
    cumsums, lowers, uppers = mp_compute_alternating_cumsums(coeffs)
    
    while True:
        if not mp_is_decreasing(coeffs):
            coeffs, cumsums, lowers, uppers = mp_increase_ks(coeffs, cumsums, lowers, uppers, psi, tau)
            continue
        
        results, complete = mp_find_smallest_ms_greater_than_U(lowers, Us)
        
        if complete:
            return results
        elif mp_more_precision_needed(uppers, Us):
            coeffs, cumsums, lowers, uppers = mp_increase_ks(coeffs, cumsums, lowers, uppers, psi, tau)
        
        coeffs, cumsums, lowers, uppers = mp_increase_ms(coeffs, cumsums, lowers, uppers, psi, tau)


class PiecewiseExponential:
    def __init__(self, slopes, intercepts):
        self.slopes = np.array(slopes, dtype=float)
        self.intercepts = np.array(intercepts, dtype=float)
        self.n_segments = len(slopes)
        
        self.exp_intercepts = np.exp(self.intercepts)
        
        if self.n_segments > 1:
            self.x_values = self._compute_boundaries(self.slopes, self.intercepts)
        else:
            self.x_values = np.array([])  # No boundaries for a single segment
        
        self.A_segments = self._compute_segment_integrals()
        self.A_total = np.sum(self.A_segments)
    
    def _compute_boundaries(self, slopes, intercepts):
        boundaries = []
        for i in range(len(intercepts) - 1):
            b = (intercepts[i+1] - intercepts[i]) / (slopes[i] - slopes[i+1])
            boundaries.append(b)
        return np.array(boundaries)
    
    def _compute_segment_integrals(self):
        A_segments = []
        for i in range(self.n_segments):
            if self.n_segments == 1:
                # Only one segment: integral from 0 to 1 (or leave as symbolic "total")
                # Here we choose integral from 0 to 1 for simplicity
                A_seg = (self.exp_intercepts[i] / self.slopes[i]) * (np.exp(self.slopes[i]) - 1)
            else:
                if i == 0:
                    A_seg = (self.exp_intercepts[i] / self.slopes[i]) * (np.exp(self.slopes[i] * self.x_values[i]) - 1)
                elif i == self.n_segments - 1:
                    x_prev = self.x_values[i-1]
                    A_seg = (-self.exp_intercepts[i] / self.slopes[i]) * np.exp(self.slopes[i] * x_prev)
                else:
                    x_curr = self.x_values[i]
                    x_prev = self.x_values[i-1]
                    A_seg = (self.exp_intercepts[i] / self.slopes[i]) * (
                        np.exp(self.slopes[i] * x_curr) - np.exp(self.slopes[i] * x_prev)
                    )
            A_segments.append(A_seg)
        return np.array(A_segments)
    
    def compute_tau(self, t):
        y = t * self.A_total
        taus = torch.zeros_like(t)
        if self.n_segments == 1:
            # Single segment: tau = (1/k) * log(1 + (k/A0) * y)
            taus = (1.0 / self.slopes[0]) * torch.log(1 + (self.slopes[0] / self.exp_intercepts[0]) * y)
        else:
            A_cumulative = np.cumsum([0] + list(self.A_segments[:-1]))
            for i in range(self.n_segments):
                if i == 0:
                    mask = y <= A_cumulative[i+1]
                elif i == self.n_segments - 1:
                    mask = y > A_cumulative[i]
                else:
                    mask = (y > A_cumulative[i]) & (y <= A_cumulative[i+1])
                
                if torch.any(mask):
                    if i == 0:
                        taus[mask] = (1.0 / self.slopes[i]) * torch.log(
                            1 + (self.slopes[i] / self.exp_intercepts[i]) * y[mask]
                        )
                    elif i == self.n_segments - 1:
                        taus[mask] = (1.0 / self.slopes[i]) * torch.log(
                            (-self.slopes[i] * (self.A_total - y[mask])) / self.exp_intercepts[i]
                        )
                    else:
                        x_prev = self.x_values[i-1]
                        taus[mask] = (1.0 / self.slopes[i]) * torch.log(
                            np.exp(self.slopes[i] * x_prev) + 
                            (self.slopes[i] / self.exp_intercepts[i]) * (y[mask] - A_cumulative[i])
                        )
        return taus
    
    def compute_beta(self, t):
        y = t * self.A_total
        betas = torch.zeros_like(t)
        if self.n_segments == 1:
            denom = self.exp_intercepts[0] + self.slopes[0] * y
            betas = self.A_total / denom
        else:
            A_cumulative = np.cumsum([0] + list(self.A_segments[:-1]))
            for i in range(self.n_segments):
                if i == 0:
                    mask = y <= A_cumulative[i+1]
                elif i == self.n_segments - 1:
                    mask = y > A_cumulative[i]
                else:
                    mask = (y > A_cumulative[i]) & (y <= A_cumulative[i+1])
                
                if torch.any(mask):
                    if i == 0:
                        denom = self.exp_intercepts[i] + self.slopes[i] * y[mask]
                        betas[mask] = self.A_total / denom
                    elif i == self.n_segments - 1:
                        denom = self.slopes[i] * (self.A_total - y[mask])
                        betas[mask] = self.A_total / denom
                    else:
                        x_prev = self.x_values[i-1]
                        B = np.exp(self.slopes[i] * x_prev) + \
                            (self.slopes[i] / self.exp_intercepts[i]) * (y[mask] - A_cumulative[i])
                        denom = self.exp_intercepts[i] * B
                        betas[mask] = self.A_total / denom
        return torch.abs(betas)
    
    def tau_to_loss(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        out = torch.empty_like(x)
        
        if self.n_segments == 1:
            out[:] = torch.exp(self.slopes[0] * x + self.intercepts[0])
        else:
            for i in range(self.n_segments):
                if i == 0:
                    mask = x < self.x_values[0]
                elif i == self.n_segments - 1:
                    mask = x >= self.x_values[-1]
                else:
                    mask = (x >= self.x_values[i-1]) & (x < self.x_values[i])
                out[mask] = torch.exp(self.slopes[i] * x[mask] + self.intercepts[i])
        return out