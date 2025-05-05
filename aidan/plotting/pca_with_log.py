import numpy as np
from sklearn.decomposition import PCA

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

# Example usage:
# Suppose X is your feature matrix (e.g., built from neuron positions, firing rates, etc.)
# results = pca_with_log_info(X, n_components=20)
# Now, results['log_explained_variance'] stores the log of the eigenvalues.
