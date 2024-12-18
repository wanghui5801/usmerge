<p align="center">
  <img src="https://ice.frostsky.com/2024/11/22/eabf1a3e6df982243482db582277b7c2.png" alt="usmerge logo" width="200"/>
</p>

# Unsupervised Merge

[![PyPI version](https://badge.fury.io/py/usmerge.svg)](https://badge.fury.io/py/usmerge)
[![Python versions](https://img.shields.io/pypi/pyversions/usmerge.svg)](https://pypi.org/project/usmerge/)
[![License](https://img.shields.io/github/license/wanghui5801/usmerge.svg)](https://github.com/wanghui5801/usmerge/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/usmerge)](https://pepy.tech/project/usmerge)
[![GitHub last commit](https://img.shields.io/github/last-commit/wanghui5801/usmerge.svg)](https://github.com/wanghui5801/usmerge/commits/main)


A simple Python package for one-dimensional data clustering, implementing various clustering algorithms including traditional and novel approaches.

## Installation

Install the package using pip:

```
pip install usmerge
```

## Features

This package provides multiple one-dimensional clustering methods:

- Equal Width Binning (equal_wid_merge)
- Equal Frequency Binning (equal_fre_merge)
- K-means Clustering (kmeans_merge)
- SOM-K Clustering (som_k_merge)
- Fuzzy C-Means (fcm_merge)
- Kernel Density Based (kernel_density_merge)
- Information Theoretic (information_merge)
- Gaussian Mixture (gaussian_mixture_merge)
- Hierarchical Density (hierarchical_density_merge)
- Jenks Natural Breaks (jenks_breaks_merge)
- Quantile-based (quantile_merge)
- DBSCAN (dbscan_1d_merge)

## Usage

### Data Format
The package accepts various input formats:
- pandas Series/DataFrame
- numpy array
- Python list/tuple
- Any iterable of numbers

### Basic Usage Examples

1. Equal Width Binning:
```python
from usmerge import equal_wid_merge
labels, edges = equal_wid_merge(data, n=3)
```

2. Equal Frequency Binning:
```python
from usmerge import equal_fre_merge
labels, edges = equal_fre_merge(data, n=3)
```

3. K-means Clustering:
```python
from usmerge import kmeans_merge
labels, edges = kmeans_merge(data, n=3, max_iter=100)
```

### Advanced Usage

1. SOM-K Clustering:
```python
from usmerge import som_k_merge
labels, edges = som_k_merge(data, n=3, sigma=0.5, learning_rate=0.5, epochs=1000)
```

2. Fuzzy C-Means:
```python
from usmerge import fcm_merge
labels, edges = fcm_merge(data, n=3, m=2.0, max_iter=100, epsilon=1e-6)
```

3. Kernel Density Based:
```python
from usmerge import kernel_density_merge
labels, edges = kernel_density_merge(data, n=3, bandwidth=None)
```

4. Jenks Natural Breaks:
```python
from usmerge import jenks_breaks_merge
labels, edges = jenks_breaks_merge(data, n=3)
```

5. Quantile-based Clustering:
```python
from usmerge import quantile_merge
labels, edges = quantile_merge(data, n=3)
```

6. DBSCAN Clustering:
```python
from usmerge import dbscan_1d_merge
labels, edges = dbscan_1d_merge(data, n=3, min_samples=3)
```

### Return Values
All clustering methods return two values:
- labels: List of cluster labels for each data point
- edges: List of cluster boundaries

## Example Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from usmerge import som_k_merge, fcm_merge, kmeans_merge, hierarchical_density_merge, dbscan_1d_merge

# Generate synthetic data with three clear clusters
np.random.seed(42)
data = np.concatenate([
    np.random.normal(0, 0.3, 50),    # First cluster
    np.random.normal(5, 0.4, 50),    # Second cluster
    np.random.normal(10, 0.3, 50)    # Third cluster
])

# Compare different clustering methods
methods = {
    'SOM-K': som_k_merge(data, n=3, sigma=0.5, learning_rate=0.5, epochs=1000),
    'FCM': fcm_merge(data, n=3, m=2.0, max_iter=100),
    'K-means': kmeans_merge(data, n=3),
    'DBSCAN': dbscan_1d_merge(data, n=3, min_samples=3),
    'Hierarchical Density': hierarchical_density_merge(data, n=3)
}

# Visualize results
plt.figure(figsize=(15, 5))
for i, (name, (labels, edges)) in enumerate(methods.items(), 1):
    plt.subplot(1, 5, i)
    plt.scatter(data, np.zeros_like(data), c=labels, cmap='viridis')
    plt.title(f'{name} Clustering')
    # Plot cluster boundaries
    for edge in edges:
        plt.axvline(x=edge, color='r', linestyle='--', alpha=0.5)
    plt.ylim(-0.5, 0.5)

plt.tight_layout()
plt.show()
```

## Parameters Guide

Each clustering method has its own set of parameters:

- SOM-K: `sigma` (neighborhood size), `learning_rate` (learning rate), `epochs` (iterations)
- FCM: `m` (fuzziness), `max_iter`, `epsilon` (convergence threshold)
- Kernel Density: `bandwidth` (kernel width)
- Information Theoretic: `alpha` (compression-accuracy trade-off)
- Gaussian Mixture: `max_iter`, `epsilon` (convergence threshold)
- Hierarchical Density: `min_cluster_size` (minimum points per cluster)
- Jenks Natural Breaks: Only requires number of clusters
- Quantile-based: Only requires number of clusters
- DBSCAN: `n` (target number of clusters), `eps` (optional neighborhood size), `min_samples` (minimum points in cluster), `max_iter` (maximum iterations for eps adjustment)

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.

## License

MIT License