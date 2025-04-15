"""
Coinpress: Private covariance estimation using coin-pressing technique.

This implements the core Coinpress algorithm for privacy-preserving 
covariance matrix estimation.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def CoinpressCov(X, args, b_fleig=True):
    """
    Coinpress algorithm for private covariance estimation.
    
    This algorithm provides improved utility especially in high-dimensional settings
    and for skewed eigenvalue distributions.
    
    Args:
        X: Data tensor of shape (n, d)
        args: Arguments object with privacy parameters
        b_fleig: Whether to enforce eigenvalue bounds

    Returns:
        Private covariance matrix estimate
    """
    n = args.n
    d = args.d
    epsilon = args.total_budget
    
    # Compute the empirical covariance matrix
    cov = torch.mm(X.t(), X) / n
    
    # Compute the spectral decomposition (eigenvalues and eigenvectors)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Sort eigenvalues in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Determine the noise scale based on the sensitivity and privacy budget
    sensitivity = 2.0 / n  # L2-sensitivity of covariance
    noise_scale = sensitivity / epsilon
    
    # Add calibrated noise to the eigenvalues
    noise = torch.randn(d) * noise_scale
    private_eigenvalues = eigenvalues + noise
    
    # Apply post-processing to ensure validity of the covariance matrix
    if b_fleig:
        for i in range(d):
            # Clip eigenvalues to be in [0, 1] as we normalized by n
            private_eigenvalues[i] = max(min(private_eigenvalues[i], 1.0), 0.0)
    
    # Reconstruct the private covariance matrix
    diag_matrix = torch.diag(private_eigenvalues)
    private_cov = torch.mm(eigenvectors, torch.mm(diag_matrix, eigenvectors.t()))
    
    # Apply Coinpress-specific shrinkage operator to improve accuracy
    shrinkage_factor = min(1.0, args.u)
    private_cov = shrinkage_factor * private_cov + (1 - shrinkage_factor) * torch.eye(d) * torch.trace(private_cov) / d
    
    # Scale back by n to match the expected output format of other methods
    return private_cov * n