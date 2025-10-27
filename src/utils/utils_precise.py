import mpmath as mp
import numpy as np
from src.utils.utils import compute_coefficient_precise, simulate_ancestral_process_precise_mp


def mp_compute_coefficients(ms, ks, psi, tau):
    precise_ms = [mp.mpf(m) for m in ms]
    precise_ks = [mp.mpf(k) for k in ks]
    
    result = mp.matrix(len(precise_ms), len(precise_ks))
    for i in range(len(precise_ms)):
        for j in range(len(precise_ks)):
            result[i, j] = compute_coefficient_precise(precise_ms[i], precise_ks[j], psi, tau)


def mp_compute_alternating_cumsums(coeffs):
    # checkerboard cum sum of rows and cols
    rows, cols = coeffs.rows, coeffs.cols
    result = mp.matrix(rows, cols)
    lowers = [mp.mpf('0')] * rows
    uppers = [mp.mpf('0')] * rows
    
    for i in range(rows):
        # precompute sign for row to avoid pow calls
        row_sign = -1 if (i % 2) else 1
        
        for j in range(cols):
            sign = row_sign if (j % 2 == 0) else -row_sign
            
            a_ij = mp.mpf(sign) * mp.mpf(coeffs[i, j])
            
            top = result[i-1, j] if i > 0 else mp.mpf('0')
            left = result[i, j-1] if j > 0 else mp.mpf('0')
            top_left = result[i-1, j-1] if (i > 0 and j > 0) else mp.mpf('0')
            result[i, j] = a_ij + top + left - top_left
            
        lowers[i] = result[i, -2]
        uppers[i] = result[i, -1]
            
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
    new_ks = np.arange(k_size, k_size + 32)
    new_ms = np.arange(m_size)
    new_coeffs = mp_compute_coefficients(new_ms, new_ks, psi, tau)
        
    # update first column of new_coeffs with last column of old coeffs
    new_cumsums = new_coeffs.copy()
    for i in range(m_size):
        sign = -1 if (i % 2) else 1
        new_cumsums[i, 0] += mp.mpf(sign) * cumsums[i, k_size-1]
    new_cumsums, lowers, uppers = mp_compute_alternating_cumsums(new_cumsums)
    
    combined_coeffs = mp_matrix_concat(coeffs, new_coeffs, axis=1)
    combined_cumsums = mp_matrix_concat(cumsums, new_cumsums, axis=1)
    
    return combined_coeffs, combined_cumsums, lowers, uppers


def mp_increase_ms(coeffs, cumsums, lowers, uppers, psi, tau):
    m_size, k_size = coeffs.rows, coeffs.cols
    new_ms = np.arange(m_size, m_size + 32)
    new_ks = np.arange(k_size)
    new_coeffs = mp_compute_coefficients(new_ms, new_ks, psi, tau)
    
    # update first row of new_coeffs with last row of old coeffs
    new_cumsums = new_coeffs.copy()
    for j in range(k_size):
        sign = -1 if (j % 2) else 1
        new_cumsums[0, j] += mp.mpf(sign) * coeffs[m_size-1, j]
    new_cumsums, new_lowers, new_uppers = mp_compute_alternating_cumsums(new_cumsums)
    
    combined_coeffs = mp_matrix_concat(coeffs, new_coeffs, axis=0)
    combined_cumsums = mp_matrix_concat(cumsums, new_cumsums, axis=0)
    combined_lowers = lowers + new_lowers
    combined_uppers = uppers + new_uppers
    
    return combined_coeffs, combined_cumsums, combined_lowers, combined_uppers


def mp_find_smallest_ms_greater_than_U(cumsums, Us):
    rows, cols = cumsums.rows, cumsums.cols
    results = [-1] * len(Us)
    for idx, U in enumerate(Us):
        for i in range(rows):
            if cumsums[i, cols-1] >= U:
                results[idx] = i
                break
    return results


def mp_find_smallest_ks_greater_than_U(lowers, Us):
    results = [-1] * len(Us)
    for idx, U in enumerate(Us):
        for i in range(len(lowers)):
            if lowers[i] >= U:
                results[idx] = i
                break
    return results


def mp_more_precision_needed(uppers, Us):
    for idx, U in enumerate(Us):
        if U < uppers[idx]:
            return True
    return False


def mp_simulate_ancestral_process(tau, psi, Us):
    mp.dps = 50  # set decimal places for high precision
    
    tau = mp.mpf(tau)
    psi = mp.mpf(psi)
    Us = [mp.mpf(U) for U in Us]
    
    start_m, start_k = 32, 64
    ms = np.arange(start_m)
    ks = np.arange(start_k)
    coeffs = mp_compute_coefficients(ms, ks, psi, tau)
    cumsums, lowers, uppers = mp_compute_alternating_cumsums(coeffs)
    
    while True:
        if mp_more_precision_needed(uppers, Us):
            coeffs, cumsums, lowers, uppers = mp_increase_ks(coeffs, cumsums, lowers, uppers, psi, tau)
        
        results = mp_find_smallest_ks_greater_than_U(lowers, Us)
        
        if np.all(results >= 0):
            return results
        else:
            coeffs, cumsums, lowers, uppers = mp_increase_ms(coeffs, cumsums, lowers, uppers, psi, tau)
            

if __name__ == "__main__":
    tau = 0.5
    psi = 4.0
    Us = [0.8, 0.95, 0.99, 0.999]
    
    print("starting")
    samples = mp_simulate_ancestral_process(tau, psi, Us)
    print("Samples:", samples)
    
    for U in Us:
        print(simulate_ancestral_process_precise_mp(tau, psi, U))
