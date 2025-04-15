# """
# Comparison experiments for different private covariance estimation algorithms.
# """

import numpy as np
import torch
import time
import sys
import os

# # Add parent directory to path to allow importing
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from config import Args
# from algorithms import EMCov, CoinpressCov, AdaptiveCovWrapper, GaussCovWrapper, SeparateCovWrapper
from utils.covariance import compute_covariance, compute_eval_evec, project
from utils.metrics import compute_diversity_relevance
from utils.data import generate_synthetic_data

import sys
import os

# Add parent directory to path to allow importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Args
from algorithms import EMCov, CoinpressCov
from algorithms.adaptive import AdaptiveCovWrapper, GaussCovWrapper, SeparateCovWrapper


def compare_methods(X, Y, epsilon_values, num_samples, dim, topk=10, num_runs=5):
    """
    Compare different methods across privacy budgets.
    
    Args:
        X: Seller dataset
        Y: Buyer dataset
        epsilon_values: List of privacy budgets to test
        num_samples: Number of samples
        dim: Dimension of the data
        topk: Number of top eigenvectors to consider
        num_runs: Number of runs for each configuration
        
    Returns:
        Dictionary with results
    """
    # Compute true covariance matrices
    cov_s = compute_covariance(X.numpy())
    cov_b = compute_covariance(Y.numpy())
    
    # Compute true eigenvalues and eigenvectors
    eval_s, evec_s = compute_eval_evec(cov_s)
    eval_b = project(cov_b, evec_s)
    
    # Compute true metrics
    true_div, true_rel = compute_diversity_relevance(eval_b, eval_s, topk)
    
    # Initialize results dictionary
    results = {
        'epsilon': epsilon_values,
        'true_div': [true_div.item()] * len(epsilon_values),
        'true_rel': [true_rel.item()] * len(epsilon_values),
        'emcov_div_mean': [],
        'emcov_div_std': [],
        'emcov_rel_mean': [],
        'emcov_rel_std': [],
        'coinpress_div_mean': [],
        'coinpress_div_std': [],
        'coinpress_rel_mean': [],
        'coinpress_rel_std': [],
        'adaptive_div_mean': [],
        'adaptive_div_std': [],
        'adaptive_rel_mean': [],
        'adaptive_rel_std': [],
        'runtime_emcov': [],
        'runtime_coinpress': [],
        'runtime_adaptive': []
    }
    
    # Run experiments for each epsilon
    for eps in epsilon_values:
        print(f"Testing with epsilon = {eps:.4f}")
        
        # Results for this epsilon
        emcov_div = []
        emcov_rel = []
        coinpress_div = []
        coinpress_rel = []
        adaptive_div = []
        adaptive_rel = []
        
        runtime_emcov_list = []
        runtime_coinpress_list = []
        runtime_adaptive_list = []
        
        # Run multiple times for statistical significance
        for run in range(num_runs):
            # Setup Args
            args = Args(total_budget=eps, delta=0, n=num_samples, d=dim, u=1.0, beta=0.1)
            
            # EMCov
            start_time = time.time()
            cov_em = EMCov(X, args, b_budget=True, b_fleig=True)
            eval_s_em, evec_s_em = compute_eval_evec(cov_em.numpy())
            eval_b_em = project(cov_b, evec_s_em)
            div_em, rel_em = compute_diversity_relevance(eval_b_em, eval_s_em, topk)
            runtime_emcov = time.time() - start_time
            
            emcov_div.append(div_em.item())
            emcov_rel.append(rel_em.item())
            runtime_emcov_list.append(runtime_emcov)
            
            # Coinpress
            start_time = time.time()
            cov_cp = CoinpressCov(X, args, b_fleig=True)
            eval_s_cp, evec_s_cp = compute_eval_evec(cov_cp.numpy())
            eval_b_cp = project(cov_b, evec_s_cp)
            div_cp, rel_cp = compute_diversity_relevance(eval_b_cp, eval_s_cp, topk)
            runtime_coinpress = time.time() - start_time
            
            coinpress_div.append(div_cp.item())
            coinpress_rel.append(rel_cp.item())
            runtime_coinpress_list.append(runtime_coinpress)
            
            # Adaptive
            start_time = time.time()
            cov_ad = AdaptiveCovWrapper(X, args, b_fleig=True)
            eval_s_ad, evec_s_ad = compute_eval_evec(cov_ad.numpy())
            eval_b_ad = project(cov_b, evec_s_ad)
            div_ad, rel_ad = compute_diversity_relevance(eval_b_ad, eval_s_ad, topk)
            runtime_adaptive = time.time() - start_time
            
            adaptive_div.append(div_ad.item())
            adaptive_rel.append(rel_ad.item())
            runtime_adaptive_list.append(runtime_adaptive)
        
        # Calculate statistics for this epsilon
        results['emcov_div_mean'].append(np.mean(emcov_div))
        results['emcov_div_std'].append(np.std(emcov_div))
        results['emcov_rel_mean'].append(np.mean(emcov_rel))
        results['emcov_rel_std'].append(np.std(emcov_rel))
        
        results['coinpress_div_mean'].append(np.mean(coinpress_div))
        results['coinpress_div_std'].append(np.std(coinpress_div))
        results['coinpress_rel_mean'].append(np.mean(coinpress_rel))
        results['coinpress_rel_std'].append(np.std(coinpress_rel))
        
        results['adaptive_div_mean'].append(np.mean(adaptive_div))
        results['adaptive_div_std'].append(np.std(adaptive_div))
        results['adaptive_rel_mean'].append(np.mean(adaptive_rel))
        results['adaptive_rel_std'].append(np.std(adaptive_rel))
        
        results['runtime_emcov'].append(np.mean(runtime_emcov_list))
        results['runtime_coinpress'].append(np.mean(runtime_coinpress_list))
        results['runtime_adaptive'].append(np.mean(runtime_adaptive_list))
        
        print(f"  EMCov:     Div = {results['emcov_div_mean'][-1]:.4f}, Rel = {results['emcov_rel_mean'][-1]:.4f}, Time = {results['runtime_emcov'][-1]:.4f}s")
        print(f"  Coinpress: Div = {results['coinpress_div_mean'][-1]:.4f}, Rel = {results['coinpress_rel_mean'][-1]:.4f}, Time = {results['runtime_coinpress'][-1]:.4f}s")
        print(f"  Adaptive:  Div = {results['adaptive_div_mean'][-1]:.4f}, Rel = {results['adaptive_rel_mean'][-1]:.4f}, Time = {results['runtime_adaptive'][-1]:.4f}s")
    
    return results


def full_exp_with_coinpress(X, Y, eps, num_samples, dim, u=1.0, method='both'):
    """
    Run a full data valuation experiment with the option to use EMCov, Coinpress, or both.

    Args:
        X, Y: Seller and buyer datasets
        eps: Privacy budget
        num_samples: Number of samples
        dim: Dimension of the data
        u: Parameter for Coinpress
        method: 'emcov', 'coinpress', or 'both'

    Returns:
        Dictionary with results for the requested method(s)
    """
    results = {}

    # Compute covariances
    cov_s = compute_covariance(X.numpy())
    cov_b = compute_covariance(Y.numpy())

    # Compute true (non-private) results
    eval_s, evec_s = compute_eval_evec(cov_s)
    eval_b = project(cov_b, evec_s)
    true_div, true_rel = compute_diversity_relevance(eval_b, eval_s, dim-2)
    results['true_div'] = true_div.item()
    results['true_rel'] = true_rel.item()

    # EMCov
    if method in ['emcov', 'both']:
        emcov_div, emcov_rel = noise_calc_emcov(cov_b, X, 10, eps, num_samples, dim)
        results['emcov_div'] = emcov_div
        results['emcov_rel'] = emcov_rel

    # Coinpress
    if method in ['coinpress', 'both']:
        coinpress_div, coinpress_rel = noise_calc_coinpress(cov_b, X, 10, eps, num_samples, dim, u)
        results['coinpress_div'] = coinpress_div
        results['coinpress_rel'] = coinpress_rel

    return results


def noise_calc_emcov(cov_b, X, steps, eps, num_samples, dim):
    """
    Calculate noise for EMCov method.
    
    Args:
        cov_b: Buyer's covariance matrix
        X: Seller's data
        steps: Number of repetitions for statistical significance
        eps: Privacy budget
        num_samples: Number of samples
        dim: Dimension
        
    Returns:
        tuple: Average diversity and relevance metrics
    """
    div_results = []
    rel_results = []

    for _ in range(steps):
        args = Args(total_budget=eps, delta=0, n=num_samples, u=1, d=dim, beta=0.1)
        estimated_cov = EMCov(X, args, b_budget=True, b_fleig=True)
        eval_s_noise, evec_s_noise = compute_eval_evec(estimated_cov.numpy())
        eval_b_noise = project(cov_b, evec_s_noise)
        noise_div, noise_rel = compute_diversity_relevance(eval_b_noise, eval_s_noise, dim-2)
        div_results.append(noise_div.item())
        rel_results.append(noise_rel.item())

    avg_div = np.mean(div_results)
    avg_rel = np.mean(rel_results)

    return avg_div, avg_rel


def noise_calc_coinpress(cov_b, X, steps, eps, num_samples, dim, u=1.0):
    """
    Calculate noise for Coinpress method.
    
    Args:
        cov_b: Buyer's covariance matrix
        X: Seller's data
        steps: Number of repetitions for statistical significance
        eps: Privacy budget
        num_samples: Number of samples
        dim: Dimension
        u: Parameter for Coinpress
        
    Returns:
        tuple: Average diversity and relevance metrics
    """
    div_results = []
    rel_results = []

    for _ in range(steps):
        args = Args(total_budget=eps, delta=0, n=num_samples, d=dim, u=u, beta=0.1)
        estimated_cov = CoinpressCov(X, args, b_fleig=True)
        eval_s_noise, evec_s_noise = compute_eval_evec(estimated_cov.numpy())
        eval_b_noise = project(cov_b, evec_s_noise)
        noise_div, noise_rel = compute_diversity_relevance(eval_b_noise, eval_s_noise, dim-2)
        div_results.append(noise_div.item())
        rel_results.append(noise_rel.item())

    avg_div = np.mean(div_results)
    avg_rel = np.mean(rel_results)

    return avg_div, avg_rel


def noise_calc_adaptive(cov_b, X, steps, eps, num_samples, dim):
    """
    Calculate noise for the adaptive method.
    
    Args:
        cov_b: Buyer's covariance matrix
        X: Seller's data
        steps: Number of repetitions for statistical significance
        eps: Privacy budget
        num_samples: Number of samples
        dim: Dimension
        
    Returns:
        tuple: Average diversity and relevance metrics
    """
    div_results = []
    rel_results = []

    # Handle potential errors with a fallback
    # try:
    for _ in range(steps):
        args = Args(total_budget=eps, delta=0, n=num_samples, d=dim, u=1, beta=0.1)
        estimated_cov = AdaptiveCovWrapper(X, args, b_fleig=True)
        eval_s_noise, evec_s_noise = compute_eval_evec(estimated_cov.numpy())
        eval_b_noise = project(cov_b, evec_s_noise)
        noise_div, noise_rel = compute_diversity_relevance(eval_b_noise, eval_s_noise, dim-2)
        div_results.append(noise_div.item())
        rel_results.append(noise_rel.item())
    # except Exception as e:
    #     print(f"Warning: Adaptive method failed with error: {e}")
    #     print("Falling back to Gaussian mechanism")
    #     # Fallback to Gaussian mechanism if adaptive fails
    #     return noise_calc_gauss(cov_b, X, steps, eps, num_samples, dim)

    avg_div = np.mean(div_results)
    avg_rel = np.mean(rel_results)

    return avg_div, avg_rel


def noise_calc_gauss(cov_b, X, steps, eps, num_samples, dim):
    """
    Calculate noise for the Gaussian mechanism (fallback for adaptive).
    
    Args:
        cov_b: Buyer's covariance matrix
        X: Seller's data
        steps: Number of repetitions for statistical significance
        eps: Privacy budget
        num_samples: Number of samples
        dim: Dimension
        
    Returns:
        tuple: Average diversity and relevance metrics
    """
    div_results = []
    rel_results = []

    for _ in range(steps):
        args = Args(total_budget=eps, delta=0, n=num_samples, d=dim, u=1, beta=0.1)
        estimated_cov = GaussCovWrapper(X, args, b_fleig=True)
        eval_s_noise, evec_s_noise = compute_eval_evec(estimated_cov.numpy())
        eval_b_noise = project(cov_b, evec_s_noise)
        noise_div, noise_rel = compute_diversity_relevance(eval_b_noise, eval_s_noise, min(dim-2, eval_s_noise.shape[0]-1))
        div_results.append(noise_div.item())
        rel_results.append(noise_rel.item())

    avg_div = np.mean(div_results)
    avg_rel = np.mean(rel_results)

    return avg_div, avg_rel


def full_exp_with_all_methods(X, Y, eps, num_samples, dim, methods=None):
    """
    Run a full data valuation experiment with multiple DP mechanisms.

    Args:
        X, Y: Seller and buyer datasets
        eps: Privacy budget
        num_samples: Number of samples
        dim: Dimension of the data
        methods: List of methods to test, options: 'emcov', 'coinpress', 'adaptive', 'gauss', 'separate'
                If None, all methods will be tested

    Returns:
        Dictionary with results for the requested methods
    """
    if methods is None:
        methods = ['emcov', 'coinpress', 'adaptive', 'gauss', 'separate']

    results = {}

    # Compute covariances
    cov_s = compute_covariance(X.numpy())
    cov_b = compute_covariance(Y.numpy())

    # Compute true (non-private) results
    eval_s, evec_s = compute_eval_evec(cov_s)
    eval_b = project(cov_b, evec_s)
    true_div, true_rel = compute_diversity_relevance(eval_b, eval_s, dim-2)
    results['true_div'] = true_div.item()
    results['true_rel'] = true_rel.item()

    # EMCov
    if 'emcov' in methods:
        emcov_div, emcov_rel = noise_calc_emcov(cov_b, X, 10, eps, num_samples, dim)
        results['emcov_div'] = emcov_div
        results['emcov_rel'] = emcov_rel

    # Coinpress
    if 'coinpress' in methods:
        coinpress_div, coinpress_rel = noise_calc_coinpress(cov_b, X, 10, eps, num_samples, dim, 1)
        results['coinpress_div'] = coinpress_div
        results['coinpress_rel'] = coinpress_rel

    # Adaptive
    if 'adaptive' in methods:
        adaptive_div, adaptive_rel = noise_calc_adaptive(cov_b, X, 10, eps, num_samples, dim)
        results['adaptive_div'] = adaptive_div
        results['adaptive_rel'] = adaptive_rel

    # Gaussian
    if 'gauss' in methods:
        args = Args(total_budget=eps, delta=0, n=num_samples, d=dim, u=1, beta=0.1)
        cov_gauss = GaussCovWrapper(X, args, b_fleig=True)
        eval_s_noise, evec_s_noise = compute_eval_evec(cov_gauss.numpy())
        eval_b_noise = project(cov_b, evec_s_noise)
        gauss_div, gauss_rel = compute_diversity_relevance(eval_b_noise, eval_s_noise, dim)
        results['gauss_div'] = gauss_div.item()
        results['gauss_rel'] = gauss_rel.item()

    # Separate
    if 'separate' in methods:
        args = Args(total_budget=eps, delta=0, n=num_samples, d=dim, u=1, beta=0.1)
        cov_separate = SeparateCovWrapper(X, args, b_fleig=True)
        eval_s_noise, evec_s_noise = compute_eval_evec(cov_separate.numpy())
        eval_b_noise = project(cov_b, evec_s_noise)
        separate_div, separate_rel = compute_diversity_relevance(eval_b_noise, eval_s_noise, dim)
        results['separate_div'] = separate_div.item()
        results['separate_rel'] = separate_rel.item()

    return results


def run_comprehensive_comparison(dim=20, num_samples=5000):
    """
    Run a comprehensive comparison of all methods across different epsilon values.

    Args:
        dim: Dimension of the data
        num_samples: Number of samples

    Returns:
        Results list and epsilon values
    """
    # Generate datasets
    torch.manual_seed(42)
    np.random.seed(42)
    X = generate_synthetic_data(dim, num_samples)
    Y = generate_synthetic_data(dim, num_samples)

    # Define epsilon values
    epsilons = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    # Run experiments for each epsilon
    results = []
    for eps in epsilons:
        print(f"\nTesting with epsilon = {eps}")
        eps_result = full_exp_with_all_methods(X, Y, eps, num_samples, dim)
        results.append(eps_result)

    # Calculate best method for each epsilon
    print("\nBest method for each epsilon:")
    for i, eps in enumerate(epsilons):
        r = results[i]
        true_div = r['true_div']
        true_rel = r['true_rel']

        method_div_errors = {}
        method_rel_errors = {}

        for method in ['emcov', 'coinpress', 'adaptive', 'gauss', 'separate']:
            if f'{method}_div' in r:
                method_div_errors[method] = abs(r[f'{method}_div'] - true_div)
                method_rel_errors[method] = abs(r[f'{method}_rel'] - true_rel)

        best_div_method = min(method_div_errors.items(), key=lambda x: x[1])[0]
        best_rel_method = min(method_rel_errors.items(), key=lambda x: x[1])[0]

        print(f"ε={eps}: Best for diversity: {best_div_method.capitalize()}, "
              f"Best for relevance: {best_rel_method.capitalize()}")

    return results, epsilons