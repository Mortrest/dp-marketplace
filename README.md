<!-- # PrivateCovariance

A framework for privacy-preserving covariance matrix estimation and data valuation.

## Overview

This repository contains implementations of several differentially private algorithms for covariance matrix estimation:

1. **EMCov**: Exponential Mechanism for private covariance estimation
2. **Coinpress**: Private covariance estimation using coin-pressing technique
3. **Adaptive**: An adaptive mechanism that chooses the best approach based on data properties
4. **Gaussian**: Standard Gaussian mechanism
5. **Separate**: Two-step approach that estimates eigenvectors and eigenvalues separately

The framework focuses on applications in private data valuation, where accurate estimation of covariance matrix structure is critical.

## Installation

```bash
# Clone the repository
git clone https://github.com/Mortrest/PrivateCovariance.git
cd PrivateCovariance

# Install requirements
pip install -r requirements.txt

# Install the package in development mode (optional)
pip install -e .
```

## Usage

### Quick Demo

To run a quick demonstration comparing different methods:

```bash
python main.py --experiment demo --dim 10 --samples 1000 --epsilon 1.0
```

### Comprehensive Comparison

To run a comprehensive comparison across different privacy budgets:

```bash
python main.py --experiment compare --dim 20 --samples 5000
```

### Performance Analysis

To analyze the performance of Coinpress compared to other methods:

```bash
python main.py --experiment performance --dim 15 --samples 1000 --epsilon 1.0
```

### Heterogeneous Data Analysis

To analyze performance on heterogeneous data types:

```bash
python main.py --experiment heterogeneous --dim 20 --samples 5000 --epsilon 1.0
```

### Run All Experiments

To run all experiments at once:

```bash
python main.py --experiment all
```

## Method Description

### EMCov

EMCov uses the exponential mechanism to privately estimate eigenvectors of the covariance matrix, along with Laplace noise for eigenvalues.

### Coinpress

Coinpress provides improved utility especially in high-dimensional settings and for skewed eigenvalue distributions. It uses a specialized noise calibration approach.

### Adaptive

The adaptive mechanism analyzes data properties and selects the best approach based on factors like dimensionality, eigenvalue distribution, and privacy budget.

## Results

Results from experiments are saved to the `results/` directory, including:

- Eigenvalue comparison plots
- Diversity and relevance metrics across privacy budgets
- Performance comparisons across dimensions
- Analysis of special data scenarios

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. -->