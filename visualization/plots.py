"""
Plotting functions for visualizing results.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_eigenvalues_comparison(eval_true, eval_em, eval_cp, dim, save_path=None):
    """
    Plot eigenvalues comparison for different methods.
    
    Args:
        eval_true: True eigenvalues
        eval_em: EMCov estimated eigenvalues
        eval_cp: Coinpress estimated eigenvalues
        dim: Dimension of the data
        save_path: Path to save the figure (if None, just displays)
    """
    plt.figure(figsize=(12, 6))
    
    # Plot eigenvalues
    plt.subplot(1, 2, 1)
    plt.plot(range(1, dim+1), eval_true, 'k-', marker='o', linewidth=2, label='True')
    plt.plot(range(1, dim+1), eval_em, 'b--', marker='x', label='EMCov')
    plt.plot(range(1, dim+1), eval_cp, 'r--', marker='s', label='Coinpress')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue Magnitude')
    plt.title('Eigenvalue Comparison')
    plt.legend()
    plt.grid(True)

    # Plot relative errors
    plt.subplot(1, 2, 2)
    rel_err_em = np.abs(eval_em - eval_true) / np.maximum(eval_true, 1e-10)
    rel_err_cp = np.abs(eval_cp - eval_true) / np.maximum(eval_true, 1e-10)
    plt.plot(range(1, dim+1), rel_err_em, 'b-', marker='x', label='EMCov Error')
    plt.plot(range(1, dim+1), rel_err_cp, 'r-', marker='s', label='Coinpress Error')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Relative Error')
    plt.title('Eigenvalue Error by Component')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return plt


def plot_method_comparison(results, epsilons, title=None, save_path=None):
    """
    Plot a comparison of all methods across different epsilon values.

    Args:
        results: List of result dictionaries for each epsilon
        epsilons: List of epsilon values
        title: Optional title for the plot
        save_path: Path to save the figure (if None, just displays)
    """
    # Extract data for plotting
    true_divs = [r['true_div'] for r in results]
    true_rels = [r['true_rel'] for r in results]

    # Prepare data for each method
    methods = ['emcov', 'coinpress', 'adaptive', 'gauss', 'separate']
    method_colors = {
        'emcov': 'blue',
        'coinpress': 'red',
        'adaptive': 'green',
        'gauss': 'purple',
        'separate': 'orange'
    }

    method_data = {}
    for method in methods:
        if f'{method}_div' in results[0]:
            method_data[method] = {
                'div': [r[f'{method}_div'] for r in results],
                'rel': [r[f'{method}_rel'] for r in results]
            }

    # Create plot
    plt.figure(figsize=(14, 12))

    # Plot diversity
    plt.subplot(2, 2, 1)
    plt.plot(epsilons, true_divs, marker='o', linestyle='-', label='True', color='black', linewidth=2)

    for method, data in method_data.items():
        plt.plot(epsilons, data['div'], marker='x', linestyle='--',
                 label=method.capitalize(), color=method_colors[method])

    plt.xscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("Diversity")
    plt.title("Diversity vs. Privacy Budget")
    plt.legend()
    plt.grid(True)

    # Plot relevance
    plt.subplot(2, 2, 2)
    plt.plot(epsilons, true_rels, marker='o', linestyle='-', label='True', color='black', linewidth=2)

    for method, data in method_data.items():
        plt.plot(epsilons, data['rel'], marker='x', linestyle='--',
                 label=method.capitalize(), color=method_colors[method])

    plt.xscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("Relevance")
    plt.title("Relevance vs. Privacy Budget")
    plt.legend()
    plt.grid(True)

    # Plot diversity error
    plt.subplot(2, 2, 3)
    for method, data in method_data.items():
        div_errors = [abs(data['div'][i] - true_divs[i]) for i in range(len(epsilons))]
        plt.plot(epsilons, div_errors, marker='x', linestyle='-',
                 label=method.capitalize(), color=method_colors[method])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("Diversity Error")
    plt.title("Diversity Error vs. Privacy Budget")
    plt.legend()
    plt.grid(True)

    # Plot relevance error
    plt.subplot(2, 2, 4)
    for method, data in method_data.items():
        rel_errors = [abs(data['rel'][i] - true_rels[i]) for i in range(len(epsilons))]
        plt.plot(epsilons, rel_errors, marker='x', linestyle='-',
                 label=method.capitalize(), color=method_colors[method])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("Relevance Error")
    plt.title("Relevance Error vs. Privacy Budget")
    plt.legend()
    plt.grid(True)

    # Set overall title if provided
    if title:
        plt.suptitle(title, fontsize=16, y=0.98)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return plt


def plot_dimension_comparison(dim_results, save_path=None):
    """
    Plot comparison of methods across different dimensions.
    
    Args:
        dim_results: Dictionary with dimension results
        save_path: Path to save the figure (if None, just displays)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dim_results['dimension'], dim_results['em_errors'], marker='o', label='EMCov Error')
    plt.plot(dim_results['dimension'], dim_results['cp_errors'], marker='s', label='Coinpress Error')
    plt.xlabel('Dimension')
    plt.ylabel('Relative Error')
    plt.title('Error Comparison by Dimension')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    return plt


def plot_full_exp_results_with_coinpress(full_exp_results, epsilons, save_path=None):
    """
    Plot the comparison of EMCov and Coinpress across epsilon values.
    
    Args:
        full_exp_results: List of experimental results
        epsilons: List of epsilon values
        save_path: Path to save the figure (if None, just displays)
    """
    true_divs = []
    emcov_divs = []
    coinpress_divs = []
    true_rels = []
    emcov_rels = []
    coinpress_rels = []

    for eps_result in full_exp_results:
        true_divs.append(eps_result['true_div'])
        true_rels.append(eps_result['true_rel'])

        if 'emcov_div' in eps_result:
            emcov_divs.append(eps_result['emcov_div'])
            emcov_rels.append(eps_result['emcov_rel'])

        if 'coinpress_div' in eps_result:
            coinpress_divs.append(eps_result['coinpress_div'])
            coinpress_rels.append(eps_result['coinpress_rel'])

    plt.figure(figsize=(12, 10))

    # Plot diversity
    plt.subplot(2, 1, 1)
    plt.plot(epsilons, true_divs, marker='o', linestyle='-', label='True Diversity', color='k')

    if emcov_divs:
        plt.plot(epsilons, emcov_divs, marker='x', linestyle='--', label='EMCov Diversity', color='b')

    if coinpress_divs:
        plt.plot(epsilons, coinpress_divs, marker='s', linestyle='--', label='Coinpress Diversity', color='r')

    plt.xscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("Diversity")
    plt.title("Diversity vs. Privacy Budget (Epsilon)")
    plt.legend()
    plt.grid(True)

    # Plot relevance
    plt.subplot(2, 1, 2)
    plt.plot(epsilons, true_rels, marker='o', linestyle='-', label='True Relevance', color='k')

    if emcov_rels:
        plt.plot(epsilons, emcov_rels, marker='x', linestyle='--', label='EMCov Relevance', color='b')

    if coinpress_rels:
        plt.plot(epsilons, coinpress_rels, marker='s', linestyle='--', label='Coinpress Relevance', color='r')

    plt.xscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("Relevance")
    plt.title("Relevance vs. Privacy Budget (Epsilon)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return plt


def plot_heterogeneous_comparison(results, methods=None, save_path=None):
    """
    Plot comparison of methods on heterogeneous data.
    
    Args:
        results: Dictionary with heterogeneous results
        methods: List of methods to include in the plot
        save_path: Path to save the figure (if None, just displays)
    """
    if methods is None:
        methods = ['emcov', 'coinpress', 'adaptive', 'gauss', 'separate']
        
    data_types = list(results.keys())
    
    # Calculate errors for each method and data type
    errors = {}
    for method in methods:
        errors[method] = {'div': [], 'rel': []}
        
    for data_type in data_types:
        data_result = results[data_type]
        true_div = data_result['true_div']
        true_rel = data_result['true_rel']
        
        for method in methods:
            if f'{method}_div' in data_result:
                div_error = abs(data_result[f'{method}_div'] - true_div)
                rel_error = abs(data_result[f'{method}_rel'] - true_rel)
                errors[method]['div'].append(div_error)
                errors[method]['rel'].append(rel_error)
    
    # Plotting
    plt.figure(figsize=(14, 8))
    
    # Plot diversity errors
    plt.subplot(1, 2, 1)
    bar_width = 0.15
    index = np.arange(len(data_types))
    
    for i, method in enumerate(methods):
        plt.bar(index + i*bar_width, errors[method]['div'], bar_width,
                label=method.capitalize())
    
    plt.xlabel('Data Type')
    plt.ylabel('Diversity Error')
    plt.title('Diversity Error by Data Type')
    plt.xticks(index + bar_width * (len(methods)-1)/2, data_types)
    plt.legend()
    
    # Plot relevance errors
    plt.subplot(1, 2, 2)
    
    for i, method in enumerate(methods):
        plt.bar(index + i*bar_width, errors[method]['rel'], bar_width,
                label=method.capitalize())
    
    plt.xlabel('Data Type')
    plt.ylabel('Relevance Error')
    plt.title('Relevance Error by Data Type')
    plt.xticks(index + bar_width * (len(methods)-1)/2, data_types)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return plt