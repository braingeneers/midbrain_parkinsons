#!/usr/bin/env python3
"""
Module: plot_sttc.py

This module provides functions for plotting STTC‐related analyses from a SpikeData object.
Functions included:
  1. plot_sttc_heatmap:
         Plot a heatmap of the (raw or filtered) STTC matrix with optional vmin/vmax,
         thresholding, and logarithmic scaling.
  2. plot_sttc_distribution:
         Plot a histogram (and optional KDE) of STTC values extracted from the upper triangle.
  3. plot_sttc_vs_distance_line:
         Compute and plot a line (scatter) plot of each neuron's average STTC versus its average distance.
  4. plot_sttc_vs_distance_violin:
         Compute and plot a violin plot of each neuron's average STTC per distance bin.
  5. plot_sttc_vs_distance_scatter:
         Compute and plot a scatter plot of each neuron's average STTC versus its average distance.
  6. plot_sttc_over_time:
         Compute a moving-window STTC (raw or filtered) and plot its evolution over time.
  7. plot_sttc_vs_distance_heatmap:
         Build a 2D heatmap where the x-axis bins are STTC values (0–1) and the y-axis bins are
         inter-neuron distances (in µm), with color reflecting the number of neuron pairs in that bin.
         
Each function assumes that the SpikeData object (sd) contains:
    - sd.train: a list of spike trains (in ms),
    - sd.length: the recording length (ms), and
    - sd.neuron_data: either a dictionary or a list with a dictionary mapping neuron IDs to metadata
      (with a 'position' key giving [x, y] coordinates in µm).

If output_path is provided, the figure is saved and then closed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter

# Import STTC functions from sttc_sd.py
from analysis.sttc_sd import compute_filtered_sttc, compute_sttc_matrix, get_upper_triangle
# Import position utilities from positions_sd.py
from analysis.position_sd import get_neuron_positions, compute_distance_matrix

# ---------------------------------------------------------------------------
def plot_sttc_heatmap(sd, matrix, output_path=None, vmin=None, vmax=None, threshold=None,
                      title=None, log_scale=False):
    """
    Plot a heatmap of the STTC matrix.
    
    If precomputed_matrix is provided, it is used directly.
    
    Parameters:
        sd : SpikeData object.
        output_path : str, optional
            Directory to save the figure.
        vmin, vmax : float, optional
            Colormap scaling values.
        threshold : float, optional
            If set, STTC values below this are set to 0.
        log_scale : bool, optional
            If True, apply logarithmic normalization.
        use_filtered : bool, optional
            If True, compute a filtered STTC matrix (if precomputed_matrix not provided).
        delt : float, optional
            STTC window parameter (ms, default 20).
        n_shuffles : int, optional
            Number of randomizations (default 10).
        randomization_method : str, optional
            "local" or "global" (default "local").
        precomputed_matrix : np.ndarray, optional
            A precomputed (filtered) STTC matrix to use.
    
    Returns:
        fig, ax
    """

    sttc_matrix = matrix
    
    if threshold is not None:
        sttc_matrix = np.where(sttc_matrix >= threshold, sttc_matrix, 0)
    
    fig, ax = plt.subplots(figsize=(10,8))
    if log_scale:
        from matplotlib.colors import LogNorm
        # LogNorm requires vmin > 0
        norm = LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(sttc_matrix, cmap='YlOrRd', norm=norm)
    else:
        im = ax.imshow(sttc_matrix, cmap='YlOrRd', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="STTC")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Neuron")
    if log_scale == True:
        ax.set_title(f"STTC Heatmap (log scale, threshold: {threshold}): {title}")
    if log_scale == False:
        ax.set_title(f"STTC Heatmap (threshold:{threshold}): {title}")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"sttc_heatmap_{title}_log{log_scale}_thresh{threshold}.png"))
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
def plot_sttc_distribution(sd, matrix, title=None, output_path=None, bins=50, kde=False):
    """
    Plot a histogram (and optional KDE) of STTC values (from the upper triangle).
    
    If precomputed_matrix is provided, it is used.
    
    Returns:
        fig, ax
    """


    sttc_matrix = matrix

    values, _ = get_upper_triangle(sttc_matrix, k=1)
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.hist(values, bins=bins, color='blue', alpha=0.7, label="STTC Distribution")
    if kde:
        # Optionally use gaussian_filter1d for a rudimentary KDE or use scipy.stats.gaussian_kde.
        xs = np.linspace(np.min(values), np.max(values), 500)
        # Here we use a simple smoothing of sorted values.
        density = gaussian_filter(np.sort(values).astype(float), sigma=2)
        ax.plot(xs, density, color='darkblue', linewidth=2, label="KDE")
    ax.set_xlabel("STTC Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"STTC Distribution: {title}")    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"sttc_distribution{title}.png"))
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
def plot_sttc_vs_distance_line(sd, matrix, output_path=None, title=None, bins_range=(0,1000), n_bins=20):
    """
    Compute and plot a line/scatter plot of each neuron's average STTC versus its average distance.
    
    If precomputed_matrix is provided, it is used.
    
    Returns:
        fig, ax
    """
    positions = get_neuron_positions(sd)
    if positions.size == 0:
        raise ValueError("No valid neuron positions found.")
    
    dist_matrix = compute_distance_matrix(positions)
    np.fill_diagonal(dist_matrix, np.nan)
    avg_distance = np.nanmean(dist_matrix, axis=1)
    
    sttc_matrix = matrix
    
    sttc_sum = np.sum(sttc_matrix, axis=1) - np.diag(sttc_matrix)
    avg_sttc = sttc_sum / (sd.N - 1)
    
    sort_idx = np.argsort(avg_distance)
    sorted_distance = avg_distance[sort_idx]
    sorted_sttc = avg_sttc[sort_idx]
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(sorted_distance, sorted_sttc, marker='o', linestyle='-', color='darkgreen')
    ax.set_xlabel("Average Distance (µm)")
    ax.set_ylabel("Average STTC")
    ax.set_title("Average STTC vs. Average Distance")
    ax.set_xlim(bins_range)
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"sttc_vs_distance_line_{title}.png"))
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
def plot_sttc_vs_distance_violin(sd, matrix, output_path=None, title=None, n_bins=5, global_bins=None):
    """
    Compute and plot a violin plot of each neuron's average STTC per distance bin.
    
    If precomputed_matrix is provided, it is used.
    
    Returns:
        fig, ax
    """
    plt.close('all')
    
    sttc_matrix = matrix
    
    N = sttc_matrix.shape[0]
    avg_sttc = np.empty(N)
    for j in range(N):
        avg_sttc[j] = np.nanmean(np.delete(sttc_matrix[j, :], j))
    
    positions_list = []
    valid_indices = []
    for j, neuron in enumerate(sd.neuron_data.values()):
        pos = neuron['position']
        if not np.allclose(pos, [0, 0]):
            positions_list.append(pos)
            valid_indices.append(j)
    positions = np.array(positions_list)
    if positions.size == 0:
        print("No valid neuron positions found.")
        return None, None
    
    avg_sttc_valid = avg_sttc[valid_indices]
    dist_matrix = np.linalg.norm(positions[:, None] - positions, axis=2)
    np.fill_diagonal(dist_matrix, np.nan)
    avg_distance = np.nanmean(dist_matrix, axis=1)
    
    if global_bins is None:
        global_bins = np.linspace(np.floor(np.nanmin(avg_distance)),
                                  np.ceil(np.nanmax(avg_distance)),
                                  n_bins + 1)
    else:
        global_bins = np.array(global_bins)
    
    bin_labels = [f"{global_bins[b]:.1f}-{global_bins[b+1]:.1f}" for b in range(n_bins)]
    sttc_bins_dict = {label: [] for label in bin_labels}
    for b in range(n_bins):
        if b < n_bins - 1:
            mask = (avg_distance >= global_bins[b]) & (avg_distance < global_bins[b+1])
        else:
            mask = (avg_distance >= global_bins[b]) & (avg_distance <= global_bins[b+1])
        if np.sum(mask) > 0:
            sttc_bins_dict[bin_labels[b]] = avg_sttc_valid[mask]
        else:
            sttc_bins_dict[bin_labels[b]] = np.array([np.nan])
    
    bin_col = []
    sttc_col = []
    for label, vals in sttc_bins_dict.items():
        bin_col.extend([label] * len(vals))
        sttc_col.extend(vals)
    df = pd.DataFrame({"Distance Bin": bin_col, "Average STTC": sttc_col})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x="Distance Bin", y="Average STTC", data=df, inner="quartile", palette="Set3", ax=ax)
    ax.set_xlabel("Average Separation Distance (µm)")
    ax.set_ylabel("Average STTC")
    ax.set_ylim(0, 1)
    ax.set_title("STTC Distribution by Distance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"sttc_vs_distance_violin_{title}.png"))
        plt.close(fig)
    
    return fig, ax

# ---------------------------------------------------------------------------
def plot_sttc_vs_distance_scatter(sd, matrix, title= None, output_path=None):
    """
    Compute and plot a scatter plot of each neuron's average STTC versus its average distance.
    
    If precomputed_matrix is provided, it is used.
    
    Returns:
        fig, ax
    """
    plt.close('all')
    
    sttc_matrix = matrix
    
    N = sttc_matrix.shape[0]
    avg_sttc = np.empty(N)
    for j in range(N):
        avg_sttc[j] = np.nanmean(np.delete(sttc_matrix[j, :], j))
    
    positions_list = []
    valid_indices = []
    for j, neuron in enumerate(sd.neuron_data.values()):
        pos = neuron['position']
        if not np.allclose(pos, [0, 0]):
            positions_list.append(pos)
            valid_indices.append(j)
    positions = np.array(positions_list)
    if positions.size == 0:
        print("No valid neuron positions found.")
        return None, None
    
    avg_sttc_valid = avg_sttc[valid_indices]
    dist_matrix = np.linalg.norm(positions[:, None] - positions, axis=2)
    np.fill_diagonal(dist_matrix, np.nan)
    avg_distance = np.nanmean(dist_matrix, axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(avg_distance, avg_sttc_valid, c='darkorange', edgecolor='k', s=60, alpha=0.8)
    ax.set_xlabel("Average Distance (µm)")
    ax.set_ylabel("Average STTC")
    ax.set_ylim(0, 1)
    ax.set_title("Average STTC vs. Average Distance")
    plt.tight_layout()
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"sttc_vs_distance_scatter_{title}.png"))
        plt.close(fig)
    
    return fig, ax

# ---------------------------------------------------------------------------
def plot_sttc_over_time(sd, output_path=None, title=None, window_size=10.0, delt=20):
    """
    Plot the evolution of STTC over time by computing a moving-window STTC.
    
    If use_filtered is True, the filtered method (difference) is used for each window.
    
    Returns:
        fig, ax
    """
    sttc_over_time = []
    time_points = []
    max_time = np.max(np.hstack(sd.train)) / 1000.0  # convert ms to seconds
    current_time = 0.0
    while current_time + window_size <= max_time:
        window_trains = []
        for train in sd.train:
            window_train = train[(train/1000.0 >= current_time) & (train/1000.0 < current_time + window_size)]
            window_trains.append(window_train)
        if len(window_trains) > 1:
            sttc_matrix = compute_sttc_matrix(window_trains, window_size*1000, delt=delt)
            vals, _ = get_upper_triangle(sttc_matrix, k=1)
            if vals.size > 0:
                sttc_over_time.append(np.nanmean(vals))
                time_points.append(current_time + window_size/2)
        current_time += window_size
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(time_points, sttc_over_time, color='blue', linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average STTC")
    ax.set_title("STTC Over Time")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"sttc_over_time_{title}.png"))
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
def plot_sttc_vs_distance_heatmap(sd, matrix, output_path=None, n_bins_distance=50, global_bins=None, 
                                  n_bins_sttc=50, heatmap_vmax=10, title=None):
    """
    Generate a 2D heatmap of neuron pair counts over STTC and distance bins.
        
    Returns:
        fig, ax
    """
    sttc_matrix = matrix
    
    positions = get_neuron_positions(sd)
    if positions.size == 0:
        raise ValueError("No valid neuron positions found.")
    
    dist_matrix = compute_distance_matrix(positions)
    triu_indices = np.triu_indices_from(sttc_matrix, k=1)
    sttc_values = sttc_matrix[triu_indices]
    distance_values = dist_matrix[triu_indices]
    
    sttc_bins = np.linspace(0, 1, n_bins_sttc + 1)
    if global_bins is None:
        global_bins = np.linspace(np.floor(np.nanmin(distance_values)),
                                  np.ceil(np.nanmax(distance_values)),
                                  n_bins_distance + 1)
    else:
        global_bins = np.array(global_bins)
    
    heatmap_matrix, _, _ = np.histogram2d(distance_values, sttc_values, bins=[global_bins, sttc_bins])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = sns.heatmap(heatmap_matrix, cmap="viridis", vmin=0, vmax=heatmap_vmax,
                     cbar_kws={'label': 'Neuron Pair Count'})
    
    num_xticks = 6
    xtick_positions = np.linspace(0.5, n_bins_sttc - 0.5, num_xticks)
    xtick_labels = ["{:.1f}".format(val) for val in np.linspace(0, 1, num_xticks)]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    
    num_yticks = 5
    ytick_positions = np.linspace(0.5, n_bins_distance - 0.5, num_yticks)
    ytick_labels = ["{:.0f}".format(val) for val in np.linspace(global_bins[0], global_bins[-1], num_yticks)]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)
    
    ax.invert_yaxis()
    ax.set_xlabel("STTC Value")
    ax.set_ylabel("Distance (µm)")
    ax.set_title(f"STTC vs. Distance: {title}")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, f"sttc_vs_distance_{title}.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------

def plot_sttc_distribution_overlay(sd, matrix, title=None, output_path=None, bins=50):
    """
    Plot an overlay histogram (and optional KDE) of two STTC distributions:
      - One computed from the sd object using compute_sttc_matrix.
      - The other provided as the input matrix.
      
    Parameters:
        sd: Data object containing neuron data.
        matrix: STTC matrix provided externally.
        title: Title for the plot.
        output_path: Directory to save the figure (if provided).
        bins: Number of bins for the histogram.
    
    Returns:
        fig, ax: The matplotlib figure and axes objects.
    """
    # Compute the STTC matrix from the sd object
    computed_matrix = sd.spike_time_tilings()
    
    # Get upper triangle values (avoiding self-comparisons) for both matrices
    computed_values, _ = get_upper_triangle(computed_matrix, k=1)
    input_values, _ = get_upper_triangle(matrix, k=1)
    
    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(12,6))
    
    # Plot histograms for both distributions
    ax.hist(computed_values, bins=bins, color='blue', alpha=0.5, label="Original STTC Distribution")
    ax.hist(input_values, bins=bins, color='red', alpha=0.5, label="Randomized STTC Distribution")
    
    ax.set_xlabel("STTC Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"STTC Distribution: {title}")
    ax.set_xlim(0, 1)
    ax.legend()
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, f"sttc_distribution_{title}.png")
        plt.savefig(fname)
        plt.close(fig)
    
    return fig, ax



# ---------------------------------------------------------------------------
# For testing purposes:
if __name__ == '__main__':
    # Example usage (for interactive testing):
    # from load_data.load_npz import load_spikedata
    # sd = load_spikedata("path/to/data.npz")
    # fig, ax = plot_sttc_heatmap(sd, output_path="output/")
    pass
