"""
Evaluation metrics for data valuation.
"""

import numpy as np


def compute_diversity_relevance(buyer_eigenvalues, seller_variances, d, terms=False):
    """
    Compute diversity and relevance metrics based on eigenvalues.
    
    These metrics are used to evaluate data valuation in a privacy-preserving
    marketplace setting.

    Args:
        buyer_eigenvalues (numpy.ndarray): Eigenvalues of buyer's covariance matrix
        seller_variances (numpy.ndarray): Eigenvalues of seller's covariance matrix
        d (int): Dimension to consider (typically dim-2)
        terms (bool): Whether to return individual terms

    Returns:
        tuple: (diversity, relevance) metrics, and optionally their component terms
    """
    # Make sure d is not larger than the available eigenvalues
    d = min(d, len(buyer_eigenvalues), len(seller_variances))
    
    # Ensure there's at least one dimension to work with
    if d <= 0:
        d = 1
    
    buyer_eigenvalues = buyer_eigenvalues[:d]
    seller_variances = seller_variances[:d]
    
    # Compute diversity
    diversity_terms = np.abs(buyer_eigenvalues - seller_variances) / np.maximum(buyer_eigenvalues, seller_variances)
    diversity = np.prod(diversity_terms) ** (1/d)

    # Compute relevance
    relevance_terms = np.minimum(buyer_eigenvalues, seller_variances) / np.maximum(buyer_eigenvalues, seller_variances)
    relevance = np.prod(relevance_terms) ** (1/d)
    
    if terms:
        return diversity, relevance, diversity_terms, relevance_terms
    else:
        return diversity, relevance