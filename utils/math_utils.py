"""
Mathematical utility functions.
"""

import numpy as np
import torch
from scipy.optimize import root_scalar


def root_bisect_dec(x0, x1, func, args, T=10, thres=1e-8):
    """
    Bisection method for finding roots of decreasing functions.
    
    Args:
        x0, x1: Initial interval bounds
        func: Function to find root of
        args: Arguments to pass to func
        T: Maximum number of iterations
        thres: Convergence threshold

    Returns:
        Root approximation
    """
    left = x0
    right = x1
    mid = 0.5 * (left + right)
    mid_val = func(mid, args)
    t = 0
    err = abs(mid_val)
    
    while t < T and err > thres:
        if mid_val > 0:
            left = mid
        else:
            right = mid
        mid = 0.5 * (left + right)
        mid_val = func(mid, args)
        err = abs(mid_val)
        t = t + 1
        
    return mid


def constr_bingham(x, Da):
    """
    Constraint function for Bingham distribution parameters.
    
    Args:
        x: Parameter value
        Da: Eigenvalues

    Returns:
        Constraint value
    """
    d = len(Da)
    farr = [1/(x + 2*Da[j]) for j in range(d)]
    f = sum(farr)
    return f - 1


def find_bingham(cov, eps, d, batch=2):
    """
    Find Bingham distribution parameters.
    
    Args:
        cov: Covariance matrix
        eps: Privacy parameter
        d: Dimension
        batch: Batch size for sampling

    Returns:
        Bingham distribution parameter
    """
    Uc, Dc, Vc = cov.svd()
    lamb_1 = max(Dc)
    Da = -eps/4.0 * (Dc - lamb_1)
    A = -eps/4.0 * cov + eps/4.0 * (lamb_1) * torch.eye(d)
    
    b = root_scalar(constr_bingham, args=Da, bracket=[1, d+1]).root
    
    ohm = torch.eye(d) + 2./b * A
    ohm_inv = torch.linalg.inv(ohm)
    logM = -0.5 * (d-b) + d/2. * np.log(d/b)
    zero_mean = torch.zeros(d)
    Z = torch.distributions.multivariate_normal.MultivariateNormal(zero_mean, ohm_inv)
    
    while True:
        z = Z.sample((batch,)).t()
        v = torch.divide(z, torch.norm(z, dim=0))
        u = torch.rand(batch)
        pr1 = torch.diag(torch.matmul(torch.matmul(v.t(), A), v))
        pr = torch.diag(torch.matmul(torch.matmul(v.t(), ohm), v))
        pr = torch.exp(-pr1 + d/2. * np.log(pr) - logM)
        success = (u < pr).squeeze()
        
        if (sum(success) > 0):
            ind = np.argmax(success > 0)
            v_out = v[:, ind]
            return v_out


def advanced_comp(x, ep0, k, delta):
    """
    Advanced composition for privacy parameters.
    
    Args:
        x: Privacy parameter
        ep0: Epsilon value
        k: Number of compositions
        delta: Delta parameter

    Returns:
        Composition result
    """
    comp = np.sqrt(2.*k*np.log(1./delta)) * x + k * x * (np.exp(x) - 1)
    return comp - ep0


def convert_eps(ep0, k, delta):
    """
    Convert epsilon parameter using advanced composition.
    
    Args:
        ep0: Original epsilon
        k: Number of compositions
        delta: Delta parameter

    Returns:
        Converted epsilon
    """
    a = k
    b = np.sqrt(2.*k*np.log(1./delta))
    c = -ep0
    r0 = (-b + np.sqrt(b*b - 4*a*c)) / (2*a)
    return r0