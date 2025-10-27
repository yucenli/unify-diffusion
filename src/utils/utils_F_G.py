import torch
import numpy as np
import scipy
import logging
import mpmath as mp
from tqdm import tqdm


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def a_k_psi_round(k, psi, t, pi_x0, x_t_x0):
    x_round = np.round(x_t_x0.cpu(), 7)
    return a_k_psi_scipy(k, psi, t.cpu(), pi_x0.cpu(), x_round).to(x_t_x0.device)


def b_k_psi_round(k, psi, t, pi_x0, x_t_x0):
    x_round = np.round(x_t_x0.cpu(), 7)
    return b_k_psi_scipy(k, psi, t.cpu(), pi_x0.cpu(), x_round).to(x_t_x0.device)


def compute_series_scipy(t, x_0, x_t, max_terms, series='G'):
    original_device = x_t.device
    t = t.double()
    x_t = x_t.double()

    if series == 'G':
        term_func = a_k_psi_scipy
    elif series == 'F':
        term_func = b_k_psi_scipy
    else:
        raise ValueError("series must be 'G' or 'F'")
    
    result = torch.ones_like(x_0, dtype=torch.float64, device='cpu')
    abs_result = torch.ones_like(x_0, dtype=torch.float64, device='cpu')

    C = x_t.shape[-1]
    pi_x0 = torch.ones_like(x_0, dtype=torch.float64) / C # assuming uniform prior
    if len(x_0.shape) == 2:
        x_t_x0 = x_t.gather(2, x_0.unsqueeze(-1)).squeeze(-1)
    elif len(x_0.shape) == 3: # shape B x D x C
        x_t_x0 = x_t.gather(2, x_0)
        
    t_cpu, pi_x0_cpu, x_t_x0_cpu = t.cpu(), pi_x0.cpu(), x_t_x0.cpu()
    
    for k in range(1, max_terms + 1):
        a_k = term_func(k, C, t_cpu, pi_x0_cpu, x_t_x0_cpu)
        term = ((-1) ** k) * a_k
        result = result + term
        abs_result = abs_result + torch.abs(term)
        
        condition = torch.abs(result / abs_result)
            
        if torch.any(condition < 1e-15):
            logger.warning(f"{series}_ψ series may be diverging, min condition is {condition.min().item():.3e}")
        
    if series == 'G' and torch.any(result + 1e-4 <= 0):
        logger.warning(f"G_ψ went negative, min value is {result.min().item():.4f}")    
        # result = torch.clamp(result, min=1e-10)
    
    return result.to(original_device)


def a_k_psi_scipy(k, psi, t, pi_x0, x_t_x0):
    original_dtype, original_device = t.dtype, t.device
    assert original_dtype == torch.float64 and original_device == torch.device('cpu')
    
    log_exp_term = (-k * (k + psi - 1) * t / 2).unsqueeze(-1)
    if len(pi_x0.shape) == 3:
        log_exp_term = log_exp_term.unsqueeze(-1)
    
    log_poch = scipy.special.gammaln(psi + k - 1) - scipy.special.gammaln(psi)
    log_fact = scipy.special.gammaln(k + 1)
    coeff_term =  (2 * k + psi - 1) * np.exp(log_poch - log_fact + log_exp_term)

    hyp_term = scipy.special.hyp2f1(-k, psi + k - 1, psi * pi_x0, x_t_x0)
    return coeff_term * hyp_term


def a_ks_psi_scipy(ks, psi, t, pi_x0, x_t_x0):
    # t: B, ks: len(ks), log_exp_term: B x 1 x 1 x len(ks)
    log_exp_term = (-ks * (ks + psi - 1) * t.unsqueeze(-1) / 2).reshape(-1, 1, 1, len(ks))
    
    # len(ks)
    log_poch = scipy.special.gammaln(psi + ks - 1) - scipy.special.gammaln(psi)
    log_fact = scipy.special.gammaln(ks + 1)
    
    coeff_term =  (2 * ks + psi - 1) * np.exp(log_poch - log_fact + log_exp_term)

    # B x D x C x len(ks)
    hyp_term = scipy.special.hyp2f1(-ks, psi + ks - 1, psi * pi_x0.unsqueeze(-1), x_t_x0.unsqueeze(-1))
    return coeff_term * hyp_term


def b_ks_psi_scipy(ks, psi, t, pi_x0, x_t_x0):    
    log_exp_term = (-ks * (ks + psi + 1) * t.unsqueeze(-1) / 2).reshape(-1, 1, 1, len(ks))
          
    log_poch = scipy.special.gammaln(psi + ks) - scipy.special.gammaln(psi)
    log_fact = scipy.special.gammaln(ks + 1)
    factorial_term = np.exp(log_poch - log_fact + log_exp_term)
    
    coeff_term = factorial_term * (2 * ks + psi + 1) * (psi + ks) / ((psi + 1) * psi)
    
    hyp_term = scipy.special.hyp2f1(-ks, psi + ks + 1, psi * pi_x0.unsqueeze(-1) + 1, x_t_x0.unsqueeze(-1))
    
    return coeff_term * hyp_term


def b_k_psi_scipy(k, psi, t, pi_x0, x_t_x0):    
    original_dtype, original_device = t.dtype, t.device
    assert original_dtype == torch.float64 and original_device == torch.device('cpu')
    
    log_exp_term = (-k * (k + psi + 1) * t / 2).unsqueeze(-1)
    if len(pi_x0.shape) == 3:
        log_exp_term = log_exp_term.unsqueeze(-1)
          
    log_poch = scipy.special.gammaln(psi + k) - scipy.special.gammaln(psi)
    log_fact = scipy.special.gammaln(k + 1)
    factorial_term = np.exp(log_poch - log_fact + log_exp_term)
    coeff_term = factorial_term * (2 * k + psi + 1) * (psi + k) / ((psi + 1) * psi)
    
    hyp_term = scipy.special.hyp2f1(-k, psi + k + 1, psi * pi_x0 + 1, x_t_x0)
    
    return coeff_term * hyp_term


def compute_T_j(j, psi, max_k, series='G'):
    T_j = mp.eye(max_k)
    if series == 'G':
        denom = mp.mpf(psi + j - 1)
    else:
        denom = mp.mpf(psi + j + 1)

    for i in range(j+1, max_k):
        if series == 'G':
            T_j[i,i] = mp.mpf(psi + i + j - 1) / denom
        else:
            T_j[i,i] = mp.mpf(psi + i + j + 1) / denom
        if i > 0:
            T_j[i,i-1] = -mp.mpf(i) / denom
    return T_j


def compute_T(psi, max_k, series='G'):
    T = mp.eye(max_k)
    for j in range(0, max_k):
        T_j = compute_T_j(j, psi, max_k, series)
        T = T_j * T
    return T
  

def compute_M_i(max_k, psi, pi_x0, x, series='G'):
    if series == 'G':
        c = mp.mpf(pi_x0) * mp.mpf(psi)
    elif series == 'F':
        c = mp.mpf(pi_x0) * mp.mpf(psi) + mp.mpf('1')
        
    M = [mp.mpf('0')] * max_k  # allocate list of mpf
    M[0] = mp.mpf('1')
    if series == 'G':
        M[1] = mp.mpf('1') - (mp.mpf(psi) - mp.mpf('1')) * mp.mpf(x) / c
    else:
        M[1] = mp.mpf('1') - (mp.mpf(psi) + mp.mpf('1')) * mp.mpf(x) / c

    for i in range(0, max_k-2):
        denom = c + i + 1
        a = (c + 2 * i + 2 - (i + mp.mpf(psi)) * mp.mpf(x))
        b = (i + 1) * (mp.mpf('1') - mp.mpf(x))
        M[i + 2] = (a / denom) * M[i + 1] - (b / denom) * M[i]
    return M


def test_compute_M():
    series = 'F'
    k = 5
    psi = 31
    i = np.arange(k)
    for x in [0.005, 0.05, 0.5]:
        if series == 'F':
            scipy_result = scipy.special.hyp2f1(-i, psi + i + 1, 2, np.ones_like(i) * x)
        else:
            scipy_result = scipy.special.hyp2f1(-i, psi + i - 1, 1, np.ones_like(i) * x)
    
        M_0 = mp.matrix(compute_M_i(k, psi, 1 / psi, x, series))
        T = compute_T(psi, k, series)
        test_result = (T * M_0)
        
        print("scipy_result", scipy_result)
        print("test_result", test_result)
    
    # assert np.allclose(scipy_result, test_result, atol=1e-5)

def save_mpmath_matrix_txt(filename, M):
    with open(filename, "w") as f:
        for i in range(M.rows):
            row_str = " ".join(str(M[i,j]) for j in range(M.cols))
            f.write(row_str + "\n")

def load_mpmath_matrix_txt(filename):
    rows = []
    with open(filename, "r") as f:
        for line in f:
            rows.append([mp.mpf(x) for x in line.strip().split()])
    return mp.matrix(rows)


def compute_series_precise(series, t, x_0, x_t, max_k):
    original_device = x_t.device
    t = t.double()
    x_t = x_t.double()

    result = torch.ones_like(x_0, dtype=torch.float64, device='cpu')

    C = x_t.shape[-1]
    pi_x0 = torch.ones_like(x_0, dtype=torch.float64) / C # assuming uniform prior
    if len(x_0.shape) == 2:
        x_t_x0 = x_t.gather(2, x_0.unsqueeze(-1)).squeeze(-1)
    elif len(x_0.shape) == 3: # shape B x D x C
        x_t_x0 = x_t.gather(2, x_0)
    B, D, C = x_t.shape
    psi = C
    
    assert max_k <= 100
    if series == 'G':
        T = load_mpmath_matrix_txt(f"pickled/T_G_{C}_100.txt")[:max_k,:max_k]
    elif series == 'F':
        T = load_mpmath_matrix_txt(f"pickled/T_F_31_100.txt")[:max_k,:max_k]
    else:
        raise ValueError("series must be 'G' or 'F'")
    
    k = mp.matrix(mp.arange(max_k))
    for batch in range(B):
        log_exp_term = mp_mul(-k, (k + psi - 1)) * t[batch].item() / 2
        
        log_poch = mp_element_wise(mp.loggamma, psi + k - 1) - mp.loggamma(psi)
        log_fact = mp_element_wise(mp.loggamma, k + 1)
        factorial_term = mp_element_wise(mp.exp, log_poch - log_fact + log_exp_term)
        
        coeff_term = mp_mul(factorial_term, 2 * k + psi - 1)
    
        for dim in range(D):
            for c in range(C):
                x_t_val = x_t_x0[batch, dim, c].item()

                M0 = compute_M_i(max_k, psi, 1/C, x_t_val, series)
                hyp_terms = T * mp.matrix(M0) # 2f1 terms for 0 to k-1
                
                terms = mp_mul(coeff_term, hyp_terms)
                series_sum = mp_alternating_sum(terms)
                result[batch, dim, c] = float(series_sum)
    return result.to(original_device)



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


def compute_series_precise_individual_optimized(t, C, x_t_val, max_k, series='G'):
    """Optimized version using vectorized operations and reduced function calls"""
    psi = mp.mpf(C)
    
    # Use numpy-style arrays instead of mpmath matrices for better performance
    k_vals = [mp.mpf(i) for i in range(max_k)]
    
    # Pre-compute common terms
    psi_mpf = mp.mpf(psi)
    t_mpf = mp.mpf(t)
    x_t_mpf = mp.mpf(x_t_val)
    
    if series == 'G':
        # Vectorized computation of log terms
        log_exp_terms = [-k * (k + psi - 1) * t_mpf / 2 for k in k_vals]
        
        # More efficient Pochhammer and factorial computation
        log_poch_terms = [mp.loggamma(psi + k - 1) - mp.loggamma(psi_mpf) for k in k_vals]
        log_fact_terms = [mp.loggamma(k + 1) for k in k_vals]
        
        # Combine logarithmic terms before exp
        log_factorial_terms = [log_poch + log_exp - log_fact 
                              for log_poch, log_exp, log_fact in 
                              zip(log_poch_terms, log_exp_terms, log_fact_terms)]
        
        factorial_terms = [mp.exp(log_term) for log_term in log_factorial_terms]
        coeff_terms = [fac_term * (2 * k + psi - 1) for fac_term, k in zip(factorial_terms, k_vals)]
        
        # Compute hypergeometric terms
        c_hyp = psi_mpf / C
        hyp_terms = [mp.hyp2f1(-k, psi + k - 1, c_hyp, x_t_mpf) for k in k_vals]
        
    else:  # series == 'F'
        # Similar optimization for F series
        log_exp_terms = [-k * (k + psi + 1) * t_mpf / 2 for k in k_vals]
        
        log_poch_terms = [mp.loggamma(psi + k) - mp.loggamma(psi_mpf) for k in k_vals]
        log_fact_terms = [mp.loggamma(k + 1) for k in k_vals]
        
        log_factorial_terms = [log_poch + log_exp - log_fact 
                              for log_poch, log_exp, log_fact in 
                              zip(log_poch_terms, log_exp_terms, log_fact_terms)]
        
        factorial_terms = [mp.exp(log_term) for log_term in log_factorial_terms]
        
        # Compute c_terms more efficiently
        c_terms = [(2 * k + psi + 1) * (psi + k) / ((psi + 1) * psi) for k in k_vals]
        coeff_terms = [fac_term * c_term for fac_term, c_term in zip(factorial_terms, c_terms)]
        
        # Compute hypergeometric terms
        c_hyp = psi_mpf / C + 1
        hyp_terms = [mp.hyp2f1(-k, psi + k + 1, c_hyp, x_t_mpf) for k in k_vals]
    
    # Final term computation and alternating sum
    terms = [coeff * hyp for coeff, hyp in zip(coeff_terms, hyp_terms)]
    
    # Optimized alternating sum
    total = mp.mpf(0)
    for i, term in enumerate(terms):
        if i % 2 == 0:
            total += term
        else:
            total -= term
    
    return float(total)

def compute_series_with_early_termination(t, C, x_t_val, max_k, series='G', tolerance=1e-15):
    """Version with early termination when terms become negligible"""
    psi = C
    psi_mpf = mp.mpf(psi)
    t_mpf = mp.mpf(t)
    x_t_mpf = mp.mpf(x_t_val)
    
    total = mp.mpf(0)
    
    for k in range(max_k):
        k_mpf = mp.mpf(k)
        
        if series == 'G':
            log_exp_term = -k_mpf * (k_mpf + psi - 1) * t_mpf / 2
            log_poch = mp.loggamma(psi + k_mpf - 1) - mp.loggamma(psi_mpf)
            log_fact = mp.loggamma(k_mpf + 1)
            
            factorial_term = mp.exp(log_poch - log_fact + log_exp_term)
            coeff_term = factorial_term * (2 * k_mpf + psi - 1)
            
            c_hyp = psi_mpf / C
            hyp_term = mp.hyp2f1(-k_mpf, psi + k_mpf - 1, c_hyp, x_t_mpf)
            
        else:
            log_exp_term = -k_mpf * (k_mpf + psi + 1) * t_mpf / 2
            log_poch = mp.loggamma(psi + k_mpf) - mp.loggamma(psi_mpf)
            log_fact = mp.loggamma(k_mpf + 1)
            
            factorial_term = mp.exp(log_poch - log_fact + log_exp_term)
            c_term = (2 * k_mpf + psi + 1) * (psi + k_mpf) / ((psi + 1) * psi)
            coeff_term = factorial_term * c_term
            
            c_hyp = psi_mpf / C + 1
            hyp_term = mp.hyp2f1(-k_mpf, psi + k_mpf + 1, c_hyp, x_t_mpf)
        
        term = coeff_term * hyp_term
        
        # Add with alternating sign
        if k % 2 == 0:
            total += term
        else:
            total -= term
        
        # Early termination check
        if k > 10 and abs(term) < tolerance * abs(total):
            break
    
    return float(total)

# if __name__ == "__main__":
#     x_t = [0.]
#     max_k = 80
#     C = 31
#     t = 0.005
    
#     for x_t_val in x_t:
#         precise_result = compute_series_precise_individual_mine(0.1, C, x_t_val, max_k, series='F')
#         optimized_result = compute_series_precise_individual_optimized(0.1, C, x_t_val, max_k, series='F')
#         early_result = compute_series_with_early_termination(0.1, C, x_t_val, max_k, series='F')
#         print(precise_result, optimized_result, early_result)
        
    # import mpmath as mp
import time
import cProfile
import pstats
from io import StringIO

# Test parameters
test_params = {
    't': 0.05,
    'C': 31.,
    'x_t_val': 0.5,
    'max_k': 50,  # Reduced for reasonable profiling time
    'series': 'G'
}

def benchmark_function(func, name, n_runs=100):
    """Benchmark a function with timing and profiling"""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {name}")
    print(f"{'='*60}")
    
    # Warm up run
    result = func(**test_params)
    print(f"Result: {result}")
    
    # Timing benchmark
    start_time = time.time()
    for _ in range(n_runs):
        func(**test_params)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_runs
    total_time = end_time - start_time
    
    print(f"Total time for {n_runs} runs: {total_time:.4f} seconds")
    print(f"Average time per call: {avg_time:.6f} seconds")
    print(f"Calls per second: {1/avg_time:.2f}")
    
    # # Profiling
    # pr = cProfile.Profile()
    # pr.enable()
    # for _ in range(10):  # Profile fewer runs for cleaner output
    #     func(**test_params)
    # pr.disable()
    
    # # Print profiling stats
    # s = StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats(20)  # Top 20 functions
    
    # print(f"\nPROFILING RESULTS (10 runs):")
    # print("-" * 60)
    # print(s.getvalue())
    
    return avg_time, result

def main():
    print("Performance Comparison: Series Computation Functions")
    print(f"Test parameters: {test_params}")
    
    # Benchmark all functions
    times = {}
    results = {}
    
    # Original function
    times['original'], results['original'] = benchmark_function(
        compute_series_precise_individual_mine, 
        "Original Implementation"
    )
    
    # Optimized function
    times['optimized'], results['optimized'] = benchmark_function(
        compute_series_precise_individual_optimized, 
        "Optimized Implementation"
    )
    
    # Early termination function
    times['early_term'], results['early_term'] = benchmark_function(
        compute_series_with_early_termination, 
        "Early Termination Implementation"
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Function':<25} {'Avg Time (s)':<15} {'Speedup':<10} {'Result':<15}")
    print("-" * 80)
    
    baseline_time = times['original']
    for name, avg_time in times.items():
        speedup = baseline_time / avg_time
        result_val = results[name]
        print(f"{name:<25} {avg_time:<15.6f} {speedup:<10.2f}x {result_val:<15.6f}")
    
    # Verify results are consistent
    print(f"\nResult verification:")
    base_result = results['original']
    for name, result in results.items():
        diff = abs(result - base_result)
        print(f"{name}: difference = {diff:.2e}")

if __name__ == "__main__":
    main()

def compute_series_mixed(series, t, x_0, x_t, max_terms):
    original_device = x_t.device
    t = t.double()
    x_t = x_t.double()

    if series == 'G':
        term_func = a_ks_psi_scipy
    elif series == 'F':
        term_func = b_ks_psi_scipy
    else:
        raise ValueError("series must be 'G' or 'F'")
    
    result = torch.ones_like(x_0, dtype=torch.float64, device='cpu')
    abs_result = torch.ones_like(x_0, dtype=torch.float64, device='cpu')

    C = x_t.shape[-1]
    pi_x0 = torch.ones_like(x_0, dtype=torch.float64) / C # assuming uniform prior
    if len(x_0.shape) == 2:
        x_t_x0 = x_t.gather(2, x_0.unsqueeze(-1)).squeeze(-1)
    elif len(x_0.shape) == 3: # shape B x D x C
        x_t_x0 = x_t.gather(2, x_0)
        
    t_cpu, pi_x0_cpu, x_t_x0_cpu = t.cpu(), pi_x0.cpu(), x_t_x0.cpu()
    ks = torch.arange(max_terms)
    
    terms = term_func(ks, C, t_cpu, pi_x0_cpu, x_t_x0_cpu) # B x D x C x max_terms
    checkerboard = torch.tensor([1 if i % 2 == 0 else -1 for i in range(max_terms)], dtype=torch.float64)
    result_terms = terms * checkerboard.to(terms.device)
    result = result_terms.sum(dim=-1)
    abs_result = terms.abs().sum(dim=-1)
    
    bad_condition = (torch.abs(result / abs_result) < 1e-15)
    
    # for low condition numbers, use high precision summation
    if torch.any(bad_condition):
        logger.warning(f"{bad_condition.sum()} {series}_ψ series may be diverging")
        ill_cond_indices = (bad_condition).nonzero(as_tuple=False)
        for idx in ill_cond_indices:
            b, d, c = idx
            result[b, d, c] = compute_series_precise_individual(t_cpu[b].item(), C, x_t_x0_cpu[b, d, c].item(), max_terms, series)
            
    return result.to(original_device)
    


def b_k_psi_scipy_(k, psi, t, pi_x0, x_t_x0):    
    log_exp_term = (-k * (k + psi + 1) * t / 2)
          
    log_poch = scipy.special.gammaln(psi + k) - scipy.special.gammaln(psi)
    log_fact = scipy.special.gammaln(k + 1)
    factorial_term = np.exp(log_poch - log_fact + log_exp_term)
    coeff_term = factorial_term * (2 * k + psi + 1) * (psi + k) / ((psi + 1) * psi)
    
    hyp_term = scipy.special.hyp2f1(-k, psi + k + 1, psi * pi_x0 + 1, x_t_x0)
    print("hyp_term", hyp_term)
    
    return coeff_term * hyp_term








# @torch.compile(mode="reduce-overhead")
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
            




# if __name__ == "__main__":
    
#     # T = compute_T(31, 100, series='F')
#     # save_mpmath_matrix_txt("pickled/T_F_31_100.txt", T)

#     torch.manual_seed(0)
#     B, D, C = 1, 24, 31
#     # x0 = torch.randint(0, C, (B, D, C)).to(device)
#     x0 = torch.arange(C).unsqueeze(0).unsqueeze(0).expand(B, D, C) # shape B x D x C
#     t = torch.rand(B).double() * 0.2 + 0.05
#     x_t = torch.rand(B, D, C).double()
#     max_terms = 80
    
#     result = compute_series_mixed('F', t, x0, x_t, max_terms)
    
#     import pdb; pdb.set_trace()
        
        
#     # TODO: finalize loss sampling
#     # implement F, not only G