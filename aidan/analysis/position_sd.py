#!/usr/bin/env python3
"""
Module: positions_sd.py

This module provides functions for extracting and analyzing neuron positions
from a SpikeData object.

Functions included:
    1. get_neuron_positions(sd, neuron_index=None)
       - Returns either all neuron positions as an (N, 2) numpy array or a single neuron's position.
    
    2. compute_distance_matrix(positions)
       - Computes the Euclidean distance between each pair of neurons.
    
    3. compute_average_distance(positions)
       - Computes the average distance for each neuron (its mean distance to all other neurons).
    
    4. compute_nearest_neighbor_distances(positions)
       - Returns the distance to the nearest neighbor for each neuron.
       
    5. compute_2d_histogram(positions, bins=50, range=None)
       - Computes a 2D histogram (density matrix) of neuron positions.
       
    6. compute_2d_kde(positions, grid_size=100, bandwidth=None, range=None)
       - Computes a kernel density estimate (KDE) over the neuron positions on a grid.
       
Note: No plotting is done here; the functions simply return the calculated matrices.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde

def get_neuron_positions(sd, neuron_index=None):
    """
    Extract neuron positions from a SpikeData object.

    This function checks if the SpikeData object's neuron_data (or neuron_attributes)
    is a dictionary in the dictionary‐of‐lists format (i.e. with keys such as "position"
    mapping to lists of values). If so, it returns the positions as a NumPy array.
    Otherwise, it falls back to iterating over the items.

    Parameters:
        sd : SpikeData
            A SpikeData object that should have either 'neuron_data' or 'neuron_attributes'.
        neuron_index : int, optional
            If provided, returns only the position for that neuron.

    Returns:
        np.ndarray or tuple:
            If neuron_index is None, returns an array of shape (N, 2) where each row is [x, y];
            otherwise, returns the (x, y) tuple for the specified neuron.
    """
    positions = None

    # First, check neuron_data
    if hasattr(sd, 'neuron_data') and sd.neuron_data:
        # If neuron_data is a dict
        if isinstance(sd.neuron_data, dict):
            # Check if it's in dictionary-of-lists format (e.g., {"position": [...], ...})
            if "position" in sd.neuron_data:
                positions = np.array(sd.neuron_data["position"])
            else:
                # Otherwise assume it's a dict of per-neuron metadata dictionaries.
                positions = []
                for key, meta in sd.neuron_data.items():
                    pos = meta.get("position", None)
                    if pos is not None:
                        positions.append(pos)
                positions = np.array(positions)
        # If neuron_data is a list, assume each element is a metadata dictionary.
        elif isinstance(sd.neuron_data, list):
            positions = []
            for meta in sd.neuron_data:
                if isinstance(meta, dict):
                    pos = meta.get("position", None)
                    if pos is not None:
                        positions.append(pos)
            positions = np.array(positions)
    # Fallback: check neuron_attributes if neuron_data is not available.
    elif hasattr(sd, 'neuron_attributes') and sd.neuron_attributes:
        if isinstance(sd.neuron_attributes, dict):
            if "position" in sd.neuron_attributes:
                positions = np.array(sd.neuron_attributes["position"])
            else:
                positions = []
                for key, meta in sd.neuron_attributes.items():
                    pos = meta.get("position", None)
                    if pos is not None:
                        positions.append(pos)
                positions = np.array(positions)
        elif isinstance(sd.neuron_attributes, list):
            positions = []
            for meta in sd.neuron_attributes:
                if isinstance(meta, dict):
                    pos = meta.get("position", None)
                    if pos is not None:
                        positions.append(pos)
            positions = np.array(positions)

    if positions is None or positions.size == 0:
        raise ValueError("No valid neuron positions found in SpikeData object.")
    
    if neuron_index is not None:
        if neuron_index < 0 or neuron_index >= positions.shape[0]:
            raise IndexError("Neuron index out of bounds.")
        return positions[neuron_index]
    
    return positions


def compute_distance_matrix(positions):
    """
    Compute the pairwise Euclidean distance matrix for neuron positions.

    Parameters:
        positions : np.ndarray
            Array of shape (N, 2) containing (x, y) positions of neurons.

    Returns:
        dist_matrix : np.ndarray
            A symmetric matrix of shape (N, N) where element [i, j] is the Euclidean distance.
    """
    return squareform(pdist(positions, metric='euclidean'))

def compute_average_distance(positions):
    """
    Compute the average distance from each neuron to all other neurons.

    Parameters:
        positions : np.ndarray
            Array of shape (N, 2) with neuron positions.

    Returns:
        avg_distances : np.ndarray
            Array of length N with the mean distance from each neuron to all others.
    """
    dist_matrix = compute_distance_matrix(positions)
    np.fill_diagonal(dist_matrix, np.nan)
    return np.nanmean(dist_matrix, axis=1)

def compute_nearest_neighbor_distances(positions):
    """
    Compute the distance to the nearest neighbor for each neuron.

    Parameters:
        positions : np.ndarray
            Array of shape (N, 2) containing neuron positions.

    Returns:
        nn_distances : np.ndarray
            Array of length N where each element is the smallest nonzero distance.
    """
    dist_matrix = compute_distance_matrix(positions)
    np.fill_diagonal(dist_matrix, np.inf)
    return np.min(dist_matrix, axis=1)

def compute_2d_histogram(positions, bins=50, range=None):
    """
    Compute a 2D histogram (density matrix) of neuron positions.

    Parameters:
        positions : np.ndarray
            Array of shape (N, 2) containing neuron positions.
        bins : int or [int, int] or [array, array], optional
            The number of bins for each dimension, or a tuple/list of bin counts or arrays.
        range : tuple, optional
            A tuple ((xmin, xmax), (ymin, ymax)). If None, determined from the data.
    
    Returns:
        hist : np.ndarray
            2D array of counts (density).
        x_edges : np.ndarray
            The bin edges for the x-axis.
        y_edges : np.ndarray
            The bin edges for the y-axis.
    """
    hist, x_edges, y_edges = np.histogram2d(positions[:, 0], positions[:, 1],
                                              bins=bins, range=range)
    return hist, x_edges, y_edges

def compute_2d_kde(positions, grid_size=100, bandwidth=None, range=None):
    """
    Compute a 2D kernel density estimate (KDE) over neuron positions.

    Parameters:
        positions : np.ndarray
            Array of shape (N, 2) containing neuron positions.
        grid_size : int, optional
            Number of grid points along each axis.
        bandwidth : float, optional
            If provided, manually set the bandwidth of the KDE. Otherwise, gaussian_kde chooses.
        range : tuple, optional
            A tuple ((xmin, xmax), (ymin, ymax)). If None, determined from data with a small margin.
    
    Returns:
        X, Y : np.ndarray
            Meshgrid arrays for the x and y coordinates.
        density : np.ndarray
            2D array of estimated density values on the grid.
    """
    # If range is not provided, determine from data with a margin.
    if range is None:
        margin = 0.05
        xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
        ymin, ymax = positions[:, 1].min(), positions[:, 1].max()
        x_margin = (xmax - xmin) * margin
        y_margin = (ymax - ymin) * margin
        range = ((xmin - x_margin, xmax + x_margin),
                 (ymin - y_margin, ymax + y_margin))
    
    (xmin, xmax), (ymin, ymax) = range
    X, Y = np.meshgrid(np.linspace(xmin, xmax, grid_size),
                       np.linspace(ymin, ymax, grid_size))
    
    # Prepare data for gaussian_kde: it expects a 2 x N array.
    values = positions.T  # shape: (2, N)
    kde = gaussian_kde(values, bw_method=bandwidth)
    
    # Evaluate KDE on the grid:
    grid_coords = np.vstack([X.ravel(), Y.ravel()])
    density = kde(grid_coords).reshape(X.shape)
    
    return X, Y, density


def pair_distances(neuron_pairs, positions):
    """
    Given a list of neuron pair tuples and an array of neuron positions,
    compute the Euclidean distance between each pair.
    
    Parameters:
      neuron_pairs : list of tuple
          Each tuple is (i, j) indicating neuron indices.
      positions : np.ndarray
          Array of shape (N, 2) with neuron positions.
          
    Returns:
      distances : list of float
          A list of distances, in the same order as neuron_pairs.
    """
    distances = []
    for pair in neuron_pairs:
        i, j = pair
        dist = np.linalg.norm(positions[i] - positions[j])
        distances.append(dist)
    return distances


# For testing purposes (if needed):
if __name__ == '__main__':
    # Example usage (assuming you have a SpikeData object 'sd'):
    # from load_npz import load_spikedata
    # sd = load_spikedata('path/to/your/auto_curated.zip')
    # positions = get_neuron_positions(sd)
    # print("Neuron positions:\n", positions)
    # hist, x_edges, y_edges = compute_2d_histogram(positions, bins=50)
    # print("2D Histogram:\n", hist)
    # X, Y, density = compute_2d_kde(positions, grid_size=100)
    # print("KDE density shape:", density.shape)
    # avg_dist = compute_average_distance(positions)
    # print("Average distances:\n", avg_dist)
    # nn_dist = compute_nearest_neighbor_distances(positions)
    # print("Nearest neighbor distances:\n", nn_dist)
    # summary = summarize_positions(sd)
    # print(summary.head())
    pass
