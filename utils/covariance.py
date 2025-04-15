"""
Covariance computation utilities.
"""

import numpy as np


def compute_covariance(data):
    """
    Compute the covariance matrix of data.

    Args:
        data (numpy.ndarray): Data matrix of shape (n_samples, d)

    Returns:
        numpy.ndarray: Covariance matrix of shape (d, d)
    """
    # Center the data
    centered_data = data - np.mean(data, axis=0)
    
    # Compute covariance matrix
    n_samples = data.shape[0]
    return (centered_data.T @ centered_data) / n_samples


def compute_eval_evec(data):
    """
    Compute the eigenvalues and eigenvectors of data or covariance matrix.

    Args:
        data (numpy.ndarray): Data matrix of shape (n_samples, d) or
                              covariance matrix of shape (d, d)

    Returns:
        tuple: (eigenvalues, eigenvectors) sorted in descending order
    """
    # If input is data, compute covariance matrix
    if len(data.shape) == 2 and data.shape[0] != data.shape[1]:
        cov_matrix = compute_covariance(data)
    else:
        cov_matrix = data

    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def project(data, directions):
    """
    Compute the variance of the data in the given directions.

    Args:
        data (numpy.ndarray): Data matrix of shape (n_samples, d) or
                              covariance matrix of shape (d, d)
        directions (numpy.ndarray): Matrix of directions of shape (d, k)

    Returns:
        numpy.ndarray: Variances in the given directions
    """
    # If input is data, compute covariance matrix
    if len(data.shape) == 2 and data.shape[0] != data.shape[1]:
        cov_matrix = compute_covariance(data)
    else:
        cov_matrix = data

    # Compute variance in each direction
    evals = np.array([
        np.linalg.norm(cov_matrix @ direction)
        for direction in directions.T
    ])

    return evals