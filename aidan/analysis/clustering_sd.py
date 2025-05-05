#!/usr/bin/env python3
"""
Module: cluster_sd.py

This module provides simplified clustering routines for neurons,
based on either spatial positions, firing rates, or a combined feature matrix.
It is intended to be lean and to leverage other modules (e.g., stats_sd.py)
for advanced dimensionality reduction (PCA) if needed.

Functions:
    1. cluster_by_position(sd, n_clusters=3)
         - Clusters neurons solely based on their (x, y) coordinates.
    2. cluster_by_firing_rate(sd, n_clusters=3)
         - Clusters neurons solely based on their mean firing rates.
    3. build_feature_matrix(sd, use_position=True, use_firing_rate=True)
         - Constructs a simple feature matrix from selected features.
    4. hierarchical_clustering(X, n_clusters=None, distance_threshold=None, method='ward')
         - Performs hierarchical clustering on a feature matrix.
         
Note:
    For PCA-based dimensionality reduction, you can import functions from your stats_sd.py module.
    This module avoids plotting and returns cluster labels and matrices only.
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster

# We assume that positions_sd.py and firing_sd.py exist in your project,
# and provide functions to extract positions and firing rates.
from positions_sd import get_neuron_positions
# For firing rates, assume SpikeData has a built-in rates() method.
# Alternatively, if you have a firing_sd.py module with a function firing_rates(sd),
# you can import it instead:
# from firing_sd import firing_rates

def cluster_by_position(sd, n_clusters=3):
    """
    Cluster neurons based solely on spatial positions.

    Parameters:
        sd : SpikeData
            A SpikeData object.
        n_clusters : int
            Number of clusters to form.

    Returns:
        labels : np.ndarray
            Array of cluster labels for each neuron.
    """
    positions = get_neuron_positions(sd)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(positions)
    return labels

def cluster_by_firing_rate(sd, n_clusters=3):
    """
    Cluster neurons based solely on their mean firing rate.

    Parameters:
        sd : SpikeData
            A SpikeData object.
        n_clusters : int
            Number of clusters to form.

    Returns:
        labels : np.ndarray
            Array of cluster labels for each neuron.
    """
    # Assume sd.rates returns firing rates in Hz.
    rates = np.array(sd.rates("Hz")).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(rates)
    return labels

def build_feature_matrix(sd, use_position=True, use_firing_rate=True):
    """
    Build a combined feature matrix for each neuron from the SpikeData object.
    Currently, supports spatial position and mean firing rate.

    Parameters:
        sd : SpikeData
            A SpikeData object.
        use_position : bool
            If True, include the (x, y) coordinates.
        use_firing_rate : bool
            If True, include the mean firing rate.
    
    Returns:
        X : np.ndarray
            A 2D feature matrix of shape (n_neurons, n_features).
    """
    features = []
    
    if use_position:
        positions = get_neuron_positions(sd)  # shape: (N, 2)
        features.append(positions)
    
    if use_firing_rate:
        rates = np.array(sd.rates("Hz")).reshape(-1, 1)
        features.append(rates)
    
    if not features:
        raise ValueError("No features selected to build the feature matrix.")
    
    X = np.hstack(features)
    return X

def hierarchical_clustering(X, n_clusters=None, distance_threshold=None, method='ward'):
    """
    Perform hierarchical clustering on a feature matrix.

    Parameters:
        X : np.ndarray
            Feature matrix (n_neurons x n_features).
        n_clusters : int, optional
            Desired number of clusters (used with criterion='maxclust').
        distance_threshold : float, optional
            Linkage distance threshold for forming clusters (used with criterion='distance').
        method : str, default='ward'
            Linkage method (e.g., 'ward', 'single', 'complete').
    
    Returns:
        labels : np.ndarray
            Array of cluster labels for each neuron.
        Z : np.ndarray
            The linkage matrix.
    
    Note:
        Either n_clusters or distance_threshold must be provided.
    """
    if (n_clusters is None) and (distance_threshold is None):
        raise ValueError("Either n_clusters or distance_threshold must be specified.")
    
    Z = linkage(X, method=method)
    
    if distance_threshold is not None:
        labels = fcluster(Z, t=distance_threshold, criterion='distance')
    else:
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    return labels, Z

# Example of how you might integrate PCA from a separate module:
# from stats_sd import reduce_dimensionality
# def reduce_features(X, n_components=2):
#     return reduce_dimensionality(X, n_components)

# End of module

if __name__ == '__main__':
    # Example usage (assuming you have a valid SpikeData object 'sd'):
    # from load_npz import load_spikedata
    # sd = load_spikedata('path/to/your/auto_curated.zip')
    #
    # Cluster based on position:
    # pos_labels = cluster_by_position(sd, n_clusters=3)
    # print("Position clustering labels:", pos_labels)
    #
    # Cluster based on firing rate:
    # rate_labels = cluster_by_firing_rate(sd, n_clusters=3)
    # print("Firing rate clustering labels:", rate_labels)
    #
    # Build a combined feature matrix:
    # X = build_feature_matrix(sd, use_position=True, use_firing_rate=True)
    # print("Combined feature matrix shape:", X.shape)
    #
    # Hierarchical clustering on the feature matrix:
    # labels, Z = hierarchical_clustering(X, n_clusters=3)
    # print("Hierarchical clustering labels:", labels)
    pass
