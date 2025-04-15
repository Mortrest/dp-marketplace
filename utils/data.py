"""
Data generation and manipulation utilities.
"""

import numpy as np
import torch


def generate_synthetic_data(d, n):
    """
    Generate synthetic data for experiments.
    
    Args:
        d (int): Dimension of the data
        n (int): Number of samples
        
    Returns:
        torch.Tensor: Generated data of shape (n, d)
    """
    # Sample X from normal distribution N(0, 1)
    X = np.random.normal(0, 1, (n, d))

    # Sample U from uniform distribution U(0, 1)
    U = np.random.uniform(0, 1, (d, d))

    # Apply U to introduce correlations between features
    X_transformed = X @ U.T

    # Normalize the data as described in the paper
    X_normalized = normalize_data(X_transformed)

    return torch.tensor(X_normalized, dtype=torch.float32)


def gen(dim, num_samples):
    """
    Generate synthetic data (compatibility wrapper).
    
    This is just a wrapper around generate_synthetic_data to match 
    the naming in legacy code.
    
    Args:
        dim (int): Dimension of the data
        num_samples (int): Number of samples
        
    Returns:
        torch.Tensor: Generated data of shape (num_samples, dim)
    """
    return generate_synthetic_data(dim, num_samples)


def normalize_data(X):
    """
    Normalize data matrix.
    
    Args:
        X (numpy.ndarray): Data matrix of shape (n, d)
        
    Returns:
        numpy.ndarray: Normalized data
    """
    # Center the data (mean 0)
    X = X - X.mean(axis=0)

    # Normalize to variance 1
    X = X / X.std(axis=0)

    # Ensure each data point has L2-norm 1
    norms = np.linalg.norm(X, axis=1)
    X = X / norms[:, np.newaxis]

    return X


def create_skewed_data(dim, num_samples, skew_factor=10):
    """
    Create dataset with skewed eigenvalue distribution.
    
    Args:
        dim (int): Dimension of the data
        num_samples (int): Number of samples
        skew_factor (float): Factor controlling eigenvalue decay
        
    Returns:
        torch.Tensor: Generated data of shape (num_samples, dim)
    """
    # Create a diagonal covariance matrix with skewed eigenvalues
    eigenvalues = np.zeros(dim)
    for i in range(dim):
        eigenvalues[i] = 1.0 / (1 + i/skew_factor)

    cov_matrix = np.diag(eigenvalues)

    # Generate data from a multivariate normal with this covariance
    np.random.seed(42)
    X = np.random.multivariate_normal(np.zeros(dim), cov_matrix, num_samples)

    # Convert to torch tensor
    return torch.from_numpy(X).float()


def create_varying_norm_data(dim, num_samples):
    """
    Create dataset with varying norms (some points far from center).
    
    Args:
        dim (int): Dimension of the data
        num_samples (int): Number of samples
        
    Returns:
        torch.Tensor: Generated data of shape (num_samples, dim)
    """
    np.random.seed(42)
    X = np.random.normal(0, 1, (num_samples, dim))

    # Make some points have much larger norms
    for i in range(0, num_samples, 10):
        X[i] = X[i] * (5 + np.random.rand())

    return torch.from_numpy(X).float()


def create_clustered_data(dim, num_samples, num_clusters=3):
    """
    Create dataset with multiple clusters.
    
    Args:
        dim (int): Dimension of the data
        num_samples (int): Number of samples
        num_clusters (int): Number of clusters to create
        
    Returns:
        torch.Tensor: Generated data of shape (num_samples, dim)
    """
    np.random.seed(42)
    X = np.zeros((num_samples, dim))

    points_per_cluster = num_samples // num_clusters

    for c in range(num_clusters):
        # Create a cluster center
        center = np.random.normal(0, 5, dim)

        # Generate points around this center
        start_idx = c * points_per_cluster
        end_idx = (c + 1) * points_per_cluster if c < num_clusters-1 else num_samples

        X[start_idx:end_idx] = np.random.normal(center, 1, (end_idx - start_idx, dim))

    return torch.from_numpy(X).float()