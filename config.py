"""
Configuration and argument handling for PrivateCovariance.
"""

class Args:
    """
    Arguments class for privacy-preserving covariance estimation algorithms.
    
    Attributes:
        total_budget (float): Total privacy budget (epsilon value)
        delta (float): Delta parameter for approximate DP (0 for pure DP)
        n (int): Number of data points
        d (int): Dimension of the data
        u (float): Parameter for Coinpress algorithm
        beta (float): Failure probability
    """
    def __init__(self, total_budget, delta, n, d, u, beta=0.1):
        self.total_budget = total_budget  # epsilon value
        self.delta = delta                # delta parameter (0 for pure DP)
        self.n = n                        # number of data points
        self.d = d                        # dimension
        self.beta = beta                  # failure probability
        self.u = u                        # parameter for Coinpress