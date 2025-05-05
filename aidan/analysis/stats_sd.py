#!/usr/bin/env python3
"""
Module: stats_sd.py

This module provides general statistical and dimensionality-reduction utilities
that can be used to analyze data extracted from a SpikeData object or any numerical array.

Functions included:
    1. reduce_dimensionality(X, n_components=2)
         - Perform PCA on the input feature matrix X and return the transformed
           data, explained variance ratio, and the PCA model.
    2. moving_average(data, window_size)
         - Compute a simple moving average of a 1D array using a uniform window.
    3. cumulative_moving_average(data)
         - Compute the cumulative moving average of a 1D array.
    4. correlation_matrix(data)
         - Compute the Pearson correlation coefficient matrix for a 2D array.
    5. rolling_std(data, window_size)
         - Compute a rolling (moving) standard deviation for a 1D array.
    6. descriptive_stats(data)
         - Compute basic descriptive statistics (mean, median, std, min, max) for a 1D array.
    7. cross_correlation(x, y)
         - Compute the cross-correlation between two 1D arrays.

No plotting is performed in this module; the functions return numerical outputs.
"""

import numpy as np
from sklearn.decomposition import PCA

def reduce_dimensionality(X, n_components=2):
    """
    Perform PCA on the input feature matrix.

    Parameters:
        X : np.ndarray
            A 2D feature matrix of shape (n_samples, n_features).
        n_components : int, optional
            The number of principal components to retain (default 2).

    Returns:
        pca_result : np.ndarray
            The transformed feature matrix (n_samples x n_components).
        explained_variance : np.ndarray
            The percentage of variance explained by each component.
        pca_model : PCA object
            The fitted PCA model.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance, pca


def pca_with_log_info(X, n_components=None, log_base=10):
    """
    Perform PCA on input X and return the transformed data, the eigenvalues,
    the explained variance ratio, and the log-transformed eigenvalues.

    Parameters:
        X : np.ndarray
            The input feature matrix (n_samples x n_features).
        n_components : int, optional
            Number of principal components to keep (default: all components).
        log_base : int or float, optional
            The base of the logarithm to use (default is 10 for log10).
    
    Returns:
        results : dict
            Dictionary containing:
              - 'pca_result': The transformed data.
              - 'explained_variance': The eigenvalues.
              - 'explained_variance_ratio': Explained variance ratio.
              - 'log_explained_variance': Log-transformed eigenvalues.
              - 'pca_model': The fitted PCA model.

    Example usage:
    Suppose X is your feature matrix (e.g., built from neuron positions, firing rates, etc.)
    results = pca_with_log_info(X, n_components=20)
    Now, results['log_explained_variance'] stores the log of the eigenvalues.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)
    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    
    if log_base == 10:
        log_eigenvalues = np.log10(eigenvalues)
    else:
        log_eigenvalues = np.log(eigenvalues)  # natural log by default if log_base not 10

    return {
        'pca_result': pca_result,
        'explained_variance': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'log_explained_variance': log_eigenvalues,
        'pca_model': pca
    }




def moving_average(data, window_size):
    """
    Compute the simple moving average of a 1D array.

    Parameters:
        data : np.ndarray
            A 1D array of numerical data.
        window_size : int
            The size of the moving window.

    Returns:
        ma : np.ndarray
            The moving average, with length len(data) - window_size + 1.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    window = np.ones(window_size) / window_size
    ma = np.convolve(data, window, mode='valid')
    return ma

def cumulative_moving_average(data):
    """
    Compute the cumulative moving average (CMA) of a 1D array.

    Parameters:
        data : np.ndarray
            A 1D array of numerical data.

    Returns:
        cma : np.ndarray
            The cumulative moving average of the input data.
    """
    cma = np.empty(len(data))
    cma[0] = data[0]
    for i in range(1, len(data)):
        cma[i] = (cma[i-1] * i + data[i]) / (i + 1)
    return cma

def correlation_matrix(data):
    """
    Compute the Pearson correlation coefficient matrix for a 2D array.

    Parameters:
        data : np.ndarray
            A 2D array where each row (or column) is a variable.

    Returns:
        corr_mat : np.ndarray
            The correlation coefficient matrix.
    """
    return np.corrcoef(data)

def rolling_std(data, window_size):
    """
    Compute the rolling (moving) standard deviation of a 1D array.

    Parameters:
        data : np.ndarray
            A 1D array of numerical data.
        window_size : int
            The size of the moving window.

    Returns:
        roll_std : np.ndarray
            The rolling standard deviation.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    roll_std = np.array([np.std(data[i:i+window_size]) for i in range(len(data)-window_size+1)])
    return roll_std

def descriptive_stats(data):
    """
    Compute basic descriptive statistics for a 1D array.

    Parameters:
        data : np.ndarray
            A 1D array of numerical data.

    Returns:
        stats : dict
            A dictionary containing mean, median, standard deviation, min, and max.
    """
    stats = {
        'mean': np.nanmean(data),
        'median': np.nanmedian(data),
        'std': np.nanstd(data),
        'min': np.nanmin(data),
        'max': np.nanmax(data)
    }
    return stats

def cross_correlation(x, y):
    """
    Compute the cross-correlation between two 1D arrays.

    Parameters:
        x, y : np.ndarray
            Two 1D arrays of equal length.

    Returns:
        corr : np.ndarray
            The cross-correlation sequence.
    """
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")
    corr = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')
    # Normalize by the product of standard deviations and length
    corr = corr / (np.std(x) * np.std(y) * len(x))
    return corr

# For testing purposes
if __name__ == '__main__':
    # Create some example data
    np.random.seed(42)
    data = np.random.randn(100)
    
    print("Original data (first 5):", data[:5])
    
    # Moving average with a window of 5
    ma = moving_average(data, window_size=5)
    print("Moving average (first 5):", ma[:5])
    
    # Cumulative moving average
    cma = cumulative_moving_average(data)
    print("Cumulative moving average (first 5):", cma[:5])
    
    # Rolling standard deviation with window of 5
    rstd = rolling_std(data, window_size=5)
    print("Rolling std (first 5):", rstd[:5])
    
    # Correlation matrix for a random 2D array (10 samples, 3 features)
    X = np.random.randn(10, 3)
    corr_mat = correlation_matrix(X)
    print("Correlation matrix:\n", corr_mat)
    
    # Basic descriptive stats
    stats = descriptive_stats(data)
    print("Descriptive stats:", stats)
    
    # Cross-correlation between two signals
    x = np.sin(np.linspace(0, 2*np.pi, 100))
    y = np.cos(np.linspace(0, 2*np.pi, 100))
    xcorr = cross_correlation(x, y)
    print("Cross-correlation (center 5 values):", xcorr[len(xcorr)//2-2:len(xcorr)//2+3])
    
    # Example PCA on a random matrix
    X_pca = np.random.randn(50, 10)
    pca_result, explained_variance, pca_model = reduce_dimensionality(X_pca, n_components=3)
    print("PCA result shape:", pca_result.shape)
    print("Explained variance:", explained_variance)
