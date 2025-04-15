"""
Wrapper functions for the adaptive and other mechanisms from the original repository.

This module contains wrapper implementations that call the original functions
from the PrivateCovariance repository.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Args class for creating new parameter objects
from config import Args

# Import the original implementations
# Note: These imports assume the repository has been cloned
from adaptive.algos import AdaptiveCov, GaussCov, SeparateCov
from adaptive.utils import gaussian_tailbound, wigner_gauss_tailbound, wigner_gauss_fnormbound


def AdaptiveCovWrapper(X, args, b_fleig=False):
    """
    Wrapper function for the adaptive mechanism to estimate covariance in a differentially private manner.

    The adaptive mechanism:
    1. Privately estimates the trace of the covariance matrix
    2. Uses SVT (Sparse Vector Technique) to determine an optimal clipping threshold
    3. Adaptively chooses between different DP mechanisms based on noise analysis
    4. Applies the selected mechanism with the determined clipping threshold

    Parameters:
    - X: Input data tensor of shape (n, d)
    - args: Arguments object with privacy parameters
    - b_fleig: Whether to enforce eigenvalue bounds

    Returns:
    - Differentially private covariance matrix estimate
    """
    # Call the AdaptiveCov function from adaptive/algos.py
    # This function already handles the adaptive selection of methods and thresholds
    cov_adaptive = AdaptiveCov(X.clone(), args)

    # Additional eigenvalue bounding if requested
    if b_fleig:
        n = args.n
        d = args.d
        D, U = torch.linalg.eigh(cov_adaptive)
        for i in range(d):
            # Clip eigenvalues to be in [0, n]
            D[i] = max(min(D[i], n), 0)
        cov_adaptive = torch.mm(U, torch.mm(D.diag_embed(), U.t()))

    return cov_adaptive


def GaussCovWrapper(X, args, b_fleig=True):
    """
    Wrapper function for the standard Gaussian mechanism to estimate covariance.

    Parameters:
    - X: Input data tensor of shape (n, d)
    - args: Arguments object with privacy parameters
    - b_fleig: Whether to enforce eigenvalue bounds

    Returns:
    - Differentially private covariance matrix estimate
    """
    n = args.n
    d = args.d
    rho = args.total_budget

    # Call the GaussCov function from adaptive/algos.py
    cov_gauss = GaussCov(X.clone(), n, d, rho, b_fleig=b_fleig)

    return cov_gauss


def SeparateCovWrapper(X, args, b_fleig=False):
    """
    Wrapper function for the separate mechanism to estimate covariance.
    The separate mechanism uses a two-step approach:
    1. First estimate the eigenvectors using the Gaussian mechanism
    2. Then estimate the eigenvalues separately

    Parameters:
    - X: Input data tensor of shape (n, d)
    - args: Arguments object with privacy parameters
    - b_fleig: Whether to enforce eigenvalue bounds

    Returns:
    - Differentially private covariance matrix estimate
    """
    n = args.n
    d = args.d
    rho = args.total_budget

    # Call the SeparateCov function from adaptive/algos.py
    cov_separate = SeparateCov(X.clone(), n, d, rho, b_fleig=b_fleig)

    return cov_separate