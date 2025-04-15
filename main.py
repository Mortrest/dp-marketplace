#!/usr/bin/env python3
"""
Main entry point for running experiments on PrivateCovariance.

This script provides a command-line interface to run the various experiments
and analyses implemented in the package.
"""

import argparse
import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import subprocess
import sys

# Clone the repository if it doesn't exist
if not os.path.exists("PrivateCovariance"):
    print("Cloning PrivateCovariance repository...")
    subprocess.run(["git", "clone", "https://github.com/Mortrest/PrivateCovariance.git"], check=True)

# Add the repository to the Python path
sys.path.append("PrivateCovariance")

# Import our modules
from config import Args
from utils.data import generate_synthetic_data
from utils.covariance import compute_covariance, compute_eval_evec, project
from utils.metrics import compute_diversity_relevance

# Import algorithms
from algorithms import EMCov, CoinpressCov
from algorithms.adaptive import AdaptiveCovWrapper, GaussCovWrapper, SeparateCovWrapper

from experiments.comparison import run_comprehensive_comparison, compare_methods
from experiments.performance import analyze_coinpress_performance
from experiments.heterogeneous import analyze_heterogeneous_data, analyze_extreme_scenarios


from visualization.plots import (
    plot_eigenvalues_comparison,
    plot_method_comparison,
    plot_dimension_comparison,
    plot_full_exp_results_with_coinpress,
    plot_heterogeneous_comparison
)


def create_output_dir():
    """Create output directory for experiment results."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_quick_demo(dim=10, num_samples=1000, epsilon=1.0):
    """
    Run a quick demonstration of the different methods.
    
    Args:
        dim: Dimension of the data
        num_samples: Number of samples
        epsilon: Privacy budget
    """
    print(f"Running quick demo (dim={dim}, n={num_samples}, ε={epsilon})")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    X = generate_synthetic_data(dim, num_samples)
    
    # Compute true covariance
    cov_true = compute_covariance(X.numpy())
    eval_true, evec_true = compute_eval_evec(cov_true)
    
    # Setup privacy parameters
    args = Args(total_budget=epsilon, delta=0, n=num_samples, d=dim, u=1.0, beta=0.1)
    
    # Run EMCov
    print("\nRunning EMCov...")
    start_time = time.time()
    cov_em = EMCov(X, args, b_budget=True, b_fleig=True)
    eval_em, evec_em = compute_eval_evec(cov_em.numpy())
    em_time = time.time() - start_time
    em_error = np.mean(np.abs(eval_em - eval_true) / np.maximum(eval_true, 1e-10))
    print(f"  Time: {em_time:.4f}s")
    print(f"  Error: {em_error:.4f}")
    
    # Run Coinpress
    print("\nRunning Coinpress...")
    start_time = time.time()
    cov_cp = CoinpressCov(X, args, b_fleig=True)
    eval_cp, evec_cp = compute_eval_evec(cov_cp.numpy())
    cp_time = time.time() - start_time
    cp_error = np.mean(np.abs(eval_cp - eval_true) / np.maximum(eval_true, 1e-10))
    print(f"  Time: {cp_time:.4f}s")
    print(f"  Error: {cp_error:.4f}")
    
    # Run Adaptive
    print("\nRunning Adaptive...")
    start_time = time.time()
    cov_ad = AdaptiveCovWrapper(X, args, b_fleig=True)
    eval_ad, evec_ad = compute_eval_evec(cov_ad.numpy())
    ad_time = time.time() - start_time
    ad_error = np.mean(np.abs(eval_ad - eval_true) / np.maximum(eval_true, 1e-10))
    print(f"  Time: {ad_time:.4f}s")
    print(f"  Error: {ad_error:.4f}")
    
    # Plot results
    output_dir = create_output_dir()
    
    # Plot eigenvalues
    plt = plot_eigenvalues_comparison(eval_true, eval_em, eval_cp, dim)
    plt.savefig(os.path.join(output_dir, 'demo_eigenvalues.png'))
    plt.close()
    
    print(f"\nResults saved to {output_dir}")
    print("\nDemo completed successfully!")


def run_all_experiments():
    """Run all experiments and save results."""
    output_dir = create_output_dir()
    print(f"Running all experiments. Results will be saved to {output_dir}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Run comprehensive comparison across epsilon values
    print("\n1. Running comprehensive comparison across epsilon values...")
    results, epsilons = run_comprehensive_comparison(dim=20, num_samples=5000)
    plt = plot_method_comparison(results, epsilons, "Method Comparison")
    plt.savefig(os.path.join(output_dir, 'epsilon_comparison.png'))
    plt.close()
    
    # 2. Analyze Coinpress performance
    print("\n2. Analyzing Coinpress performance...")
    perf_results = analyze_coinpress_performance(dim=15, num_samples=1000, eps=1.0)
    plt = plot_dimension_comparison(perf_results['dimension_results'])
    plt.savefig(os.path.join(output_dir, 'dimension_comparison.png'))
    plt.close()
    
    # 3. Analyze heterogeneous data
    print("\n3. Analyzing heterogeneous data...")
    hetero_results = analyze_heterogeneous_data(dim=20, num_samples=5000, epsilon=1.0)
    plt = plot_heterogeneous_comparison(hetero_results)
    plt.savefig(os.path.join(output_dir, 'heterogeneous_comparison.png'))
    plt.close()
    
    # 4. Analyze extreme scenarios
    print("\n4. Analyzing extreme scenarios...")
    extreme_results = analyze_extreme_scenarios(num_samples=1000, epsilon=1.0)
    
    print(f"\nAll experiments completed. Results saved to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='PrivateCovariance experiments')
    
    parser.add_argument('--experiment', type=str, default='demo',
                        choices=['demo', 'compare', 'performance', 'heterogeneous', 'all'],
                        help='Which experiment to run')
    
    parser.add_argument('--dim', type=int, default=20,
                        help='Dimension of the data')
    
    parser.add_argument('--samples', type=int, default=5000,
                        help='Number of samples')
    
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Privacy budget')
    
    args = parser.parse_args()
    
    if args.experiment == 'demo':
        run_quick_demo(args.dim, args.samples, args.epsilon)
    
    elif args.experiment == 'compare':
        print(f"Running comparison experiment (dim={args.dim}, n={args.samples})")
        results, epsilons = run_comprehensive_comparison(args.dim, args.samples)
        
        output_dir = create_output_dir()
        plt = plot_method_comparison(results, epsilons, "Method Comparison")
        plt.savefig(os.path.join(output_dir, 'epsilon_comparison.png'))
        plt.close()
        
        print(f"Results saved to {output_dir}")
    
    elif args.experiment == 'performance':
        print(f"Running performance analysis (dim={args.dim}, n={args.samples}, ε={args.epsilon})")
        results = analyze_coinpress_performance(args.dim, args.samples, args.epsilon)
        
        output_dir = create_output_dir()
        plt = plot_dimension_comparison(results['dimension_results'])
        plt.savefig(os.path.join(output_dir, 'dimension_comparison.png'))
        plt.close()
        
        print(f"Results saved to {output_dir}")
    
    elif args.experiment == 'heterogeneous':
        print(f"Running heterogeneous data analysis (dim={args.dim}, n={args.samples}, ε={args.epsilon})")
        results = analyze_heterogeneous_data(args.dim, args.samples, args.epsilon)
        
        output_dir = create_output_dir()
        plt = plot_heterogeneous_comparison(results)
        plt.savefig(os.path.join(output_dir, 'heterogeneous_comparison.png'))
        plt.close()
        
        print(f"Results saved to {output_dir}")
    
    elif args.experiment == 'all':
        run_all_experiments()


if __name__ == '__main__':
    main()