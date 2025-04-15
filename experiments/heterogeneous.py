"""
Experiments with heterogeneous data for private covariance estimation.
"""

import numpy as np
import torch
import sys
import os

# Add parent directory to path to allow importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Args
from algorithms import EMCov, CoinpressCov
from algorithms.adaptive import AdaptiveCovWrapper, GaussCovWrapper, SeparateCovWrapper
from utils.covariance import compute_covariance, compute_eval_evec, project
from utils.metrics import compute_diversity_relevance
from utils.data import create_skewed_data, create_varying_norm_data, create_clustered_data
from experiments.comparison import full_exp_with_all_methods


def analyze_heterogeneous_data(dim=20, num_samples=5000, epsilon=1.0):
    """
    Create and analyze heterogeneous datasets where the adaptive method might excel.

    The adaptive method is particularly good at handling datasets with:
    1. Non-uniform eigenvalue distributions
    2. Varying norms across data points
    3. Multiple data clusters

    This function creates such datasets and compares all methods.

    Args:
        dim: Dimension of the data
        num_samples: Number of samples
        epsilon: Privacy budget

    Returns:
        Results dictionary comparing all methods
    """
    print("Analyzing heterogeneous data scenarios...")

    # Run comparison on each dataset type
    results = {}

    # 1. Skewed eigenvalue distribution
    X_skewed = create_skewed_data(dim, num_samples)
    Y_skewed = create_skewed_data(dim, num_samples)

    print("\n1. Testing on skewed eigenvalue data...")
    skewed_results = full_exp_with_all_methods(X_skewed, Y_skewed, epsilon, num_samples, dim)
    results['skewed'] = skewed_results

    # 2. Varying norm data
    X_varying = create_varying_norm_data(dim, num_samples)
    Y_varying = create_varying_norm_data(dim, num_samples)

    print("\n2. Testing on varying norm data...")
    varying_results = full_exp_with_all_methods(X_varying, Y_varying, epsilon, num_samples, dim)
    results['varying'] = varying_results

    # 3. Clustered data
    X_clustered = create_clustered_data(dim, num_samples)
    Y_clustered = create_clustered_data(dim, num_samples)

    print("\n3. Testing on clustered data...")
    clustered_results = full_exp_with_all_methods(X_clustered, Y_clustered, epsilon, num_samples, dim)
    results['clustered'] = clustered_results

    # Print summary of results
    print("\nSummary of results on heterogeneous data:")

    for data_type, data_results in results.items():
        print(f"\n{data_type.capitalize()} data:")
        true_div = data_results['true_div']
        true_rel = data_results['true_rel']

        print(f"  True: Div={true_div:.4f}, Rel={true_rel:.4f}")

        method_errors = {}
        for method in ['emcov', 'coinpress', 'adaptive', 'gauss', 'separate']:
            if f'{method}_div' in data_results:
                div_error = abs(data_results[f'{method}_div'] - true_div)
                rel_error = abs(data_results[f'{method}_rel'] - true_rel)
                method_errors[method] = (div_error, rel_error)
                print(f"  {method.capitalize()}: "
                      f"Div={data_results[f'{method}_div']:.4f} (err={div_error:.4f}), "
                      f"Rel={data_results[f'{method}_rel']:.4f} (err={rel_error:.4f})")

        # Find best method for each metric
        best_div_method = min(method_errors.items(), key=lambda x: x[1][0])[0]
        best_rel_method = min(method_errors.items(), key=lambda x: x[1][1])[0]

        print(f"  Best for diversity: {best_div_method.capitalize()}")
        print(f"  Best for relevance: {best_rel_method.capitalize()}")

    return results


def compare_adaptive_performance(dim_range=[10, 20, 30, 50], num_samples=5000, epsilon=1.0):
    """
    Compare the performance of the adaptive method vs others under different dimensions.
    
    The adaptive method should excel in high dimensions with skewed eigenvalue distributions.
    
    Args:
        dim_range: List of dimensions to test
        num_samples: Number of samples
        epsilon: Privacy budget
        
    Returns:
        Dictionary with results for each dimension
    """
    print("Comparing adaptive method performance across dimensions...")
    
    results = {}
    
    for dim in dim_range:
        print(f"\nTesting with dimension = {dim}")
        
        # Generate skewed data (which should benefit the adaptive method)
        X_skewed = create_skewed_data(dim, num_samples, skew_factor=5)
        Y_skewed = create_skewed_data(dim, num_samples, skew_factor=5)
        
        # Run all methods
        dim_results = full_exp_with_all_methods(X_skewed, Y_skewed, epsilon, num_samples, dim)
        results[dim] = dim_results
        
        # Print summary for this dimension
        true_div = dim_results['true_div']
        true_rel = dim_results['true_rel']
        
        print(f"  True: Div={true_div:.4f}, Rel={true_rel:.4f}")
        
        method_errors = {}
        for method in ['emcov', 'coinpress', 'adaptive', 'gauss', 'separate']:
            if f'{method}_div' in dim_results:
                div_error = abs(dim_results[f'{method}_div'] - true_div)
                rel_error = abs(dim_results[f'{method}_rel'] - true_rel)
                method_errors[method] = (div_error, rel_error)
                print(f"  {method.capitalize()}: "
                      f"Div={dim_results[f'{method}_div']:.4f} (err={div_error:.4f}), "
                      f"Rel={dim_results[f'{method}_rel']:.4f} (err={rel_error:.4f})")
        
        # Find best method for each metric
        best_div_method = min(method_errors.items(), key=lambda x: x[1][0])[0]
        best_rel_method = min(method_errors.items(), key=lambda x: x[1][1])[0]
        
        print(f"  Best for diversity: {best_div_method.capitalize()}")
        print(f"  Best for relevance: {best_rel_method.capitalize()}")
    
    # Calculate adaptive method's relative performance across dimensions
    adaptive_performance = {}
    for dim in dim_range:
        adaptive_performance[dim] = {}
        dim_results = results[dim]
        true_div = dim_results['true_div']
        
        # Calculate error ratios (lower is better for adaptive)
        for method in ['emcov', 'coinpress', 'gauss', 'separate']:
            if f'{method}_div' in dim_results and 'adaptive_div' in dim_results:
                method_error = abs(dim_results[f'{method}_div'] - true_div)
                adaptive_error = abs(dim_results['adaptive_div'] - true_div)
                
                # Ratio > 1 means adaptive is better
                ratio = method_error / max(adaptive_error, 1e-10)
                adaptive_performance[dim][method] = ratio
    
    # Print summary of adaptive method's improvement across dimensions
    print("\nAdaptive method performance relative to other methods (ratio > 1 means adaptive is better):")
    for dim in dim_range:
        print(f"\nDimension = {dim}:")
        for method, ratio in adaptive_performance[dim].items():
            print(f"  vs {method.capitalize()}: {ratio:.2f}x")
    
    return results, adaptive_performance


def analyze_extreme_scenarios(num_samples=1000, epsilon=1.0):
    """
    Analyze performance in extreme scenarios to identify when each method excels.
    
    This tests:
    1. Very high dimensions with few samples
    2. Extremely skewed eigenvalues
    3. Heavy outliers
    
    Args:
        num_samples: Number of samples
        epsilon: Privacy budget
        
    Returns:
        Dictionary with results for each scenario
    """
    print("Analyzing extreme scenarios...")
    
    results = {}
    
    # 1. High dimension, few samples (d > n/10 regime)
    dim_high = num_samples // 5
    print(f"\n1. High dimension scenario: d={dim_high}, n={num_samples}")
    
    X_high = create_skewed_data(dim_high, num_samples)
    Y_high = create_skewed_data(dim_high, num_samples)
    
    high_dim_results = full_exp_with_all_methods(X_high, Y_high, epsilon, num_samples, dim_high)
    results['high_dim'] = high_dim_results
    
    # 2. Extremely skewed eigenvalues
    dim_skewed = 20
    print(f"\n2. Extremely skewed eigenvalues: d={dim_skewed}, n={num_samples}")
    
    # Create data with very skewed eigenvalues (skew_factor=1 makes first few dominate)
    X_skewed = create_skewed_data(dim_skewed, num_samples, skew_factor=1)
    Y_skewed = create_skewed_data(dim_skewed, num_samples, skew_factor=1)
    
    skewed_results = full_exp_with_all_methods(X_skewed, Y_skewed, epsilon, num_samples, dim_skewed)
    results['extremely_skewed'] = skewed_results
    
    # 3. Heavy outliers
    dim_outlier = 20
    print(f"\n3. Heavy outliers scenario: d={dim_outlier}, n={num_samples}")
    
    # Create varying norm data with severe outliers
    X_outlier = create_varying_norm_data(dim_outlier, num_samples)
    # Make some outliers even more extreme
    for i in range(0, num_samples, 50):
        X_outlier[i] = X_outlier[i] * 20  # extremely large outliers
    
    Y_outlier = create_varying_norm_data(dim_outlier, num_samples)
    
    outlier_results = full_exp_with_all_methods(X_outlier, Y_outlier, epsilon, num_samples, dim_outlier)
    results['heavy_outliers'] = outlier_results
    
    # Print summary for each scenario
    for scenario, scenario_results in results.items():
        print(f"\nScenario: {scenario}")
        true_div = scenario_results['true_div']
        
        method_errors = {}
        for method in ['emcov', 'coinpress', 'adaptive', 'gauss', 'separate']:
            if f'{method}_div' in scenario_results:
                error = abs(scenario_results[f'{method}_div'] - true_div)
                method_errors[method] = error
                print(f"  {method.capitalize()}: error={error:.4f}")
        
        # Find best method
        best_method = min(method_errors.items(), key=lambda x: x[1])[0]
        print(f"  Best method: {best_method.capitalize()}")
    
    return results