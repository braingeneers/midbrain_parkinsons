#!/usr/bin/env python3
"""
Module: plot_footprints.py

This module provides footprint‐based plotting functions for a SpikeData object.
Functions include:

    1. raw_footprint: Plot a basic scatter plot of neuron positions.
    2. overlay_fr_footprint: Plot a footprint with marker sizes scaled by firing rate.
    3. plot_neuron_status: Plot an overlay of duplicate neuron positions across datasets.
    4. plot_sttc_layout: Plot a network‐style layout of neuron positions. Here, the
       filtered STTC matrix (computed using a "difference" method) is used, and only
       connections with values above a threshold are drawn.
       
Each function returns (fig, ax). If an output_path is provided, the figure is saved
to that directory and then closed.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from analysis.analysis.sttc_sd import compute_filtered_sttc
from analysis.analysis.positions_sd import get_neuron_positions, compute_average_distance, compute_nearest_neighbor_distances
# ---------------------------------------------------------------------------
def raw_footprint(sd, output_path=None):
    """
    Plot a basic footprint of neuron positions.

    Parameters:
        sd : SpikeData
            A SpikeData object with neuron positions stored in sd.neuron_data.
        output_path : str, optional
            Directory in which to save the plot. If None, the plot is returned.

    Returns:
        fig, ax : Matplotlib figure and axes objects.
    """
    # Use the first element if sd.neuron_data is a list.
    data_source = sd.neuron_data[0] if isinstance(sd.neuron_data, list) else sd.neuron_data
    positions = []
    for neuron in data_source.values():
        pos = neuron.get('position', None)
        if pos is not None and not np.allclose(pos, [0,0]):
            positions.append(pos)
    positions = np.array(positions)
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=50)
    ax.set_xlabel("Horizontal Position (µm)")
    ax.set_ylabel("Vertical Position (µm)")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, "raw_footprint.png"))
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
def overlay_fr_footprint(sd, output_path=None):
    """
    Plot a footprint with an overlay of firing rate.
    Marker size is scaled by the neuron's firing rate.

    Parameters:
        sd : SpikeData
            A SpikeData object with neuron positions (sd.neuron_data) and firing rates
            (sd.firing_rates_list[0]).
        output_path : str, optional
            Directory to save the plot. If None, the plot is returned.

    Returns:
        fig, ax : Matplotlib figure and axes objects.
    """
    data_source = sd.neuron_data[0] if isinstance(sd.neuron_data, list) else sd.neuron_data
    neuron_x, neuron_y, frates = [], [], []
    for j, neuron in enumerate(data_source.values()):
        pos = neuron.get('position', None)
        if pos is not None and not np.allclose(pos, [0,0]):
            neuron_x.append(pos[0])
            neuron_y.append(pos[1])
            frates.append(sd.firing_rates_list[0][j])
    neuron_x = np.array(neuron_x)
    neuron_y = np.array(neuron_y)
    frates = np.array(frates)
    
    fig, ax = plt.subplots(figsize=(11,9))
    ax.scatter(neuron_x, neuron_y, s=frates * 300, c='red', alpha=0.4, edgecolors='none')
    ax.set_xlabel("Horizontal Position (µm)")
    ax.set_ylabel("Vertical Position (µm)")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, "overlay_fr_footprint.png"))
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
def plot_neuron_status(neuron_data_list, cleaned_names, output_path=None):
    """
    Generate an overlay plot showing duplicate neuron positions across datasets.
    For each dataset, positions that appear in more than one dataset are plotted.

    Parameters:
        neuron_data_list : list
            A list of neuron_data dictionaries (one per dataset).
        cleaned_names : list of str
            List of cleaned dataset names.
        output_path : str, optional
            Directory to save the plot.

    Returns:
        fig, ax : Matplotlib figure and axes objects.
    """
    from collections import defaultdict
    position_map = defaultdict(list)
    for day_index, nd in enumerate(neuron_data_list):
        for neuron_id, meta in nd.items():
            pos = meta.get('position', None)
            if pos is not None:
                position_map[tuple(pos)].append(day_index)
    duplicate_positions = {pos: days for pos, days in position_map.items() if len(days) > 1}
    positions = np.array(list(duplicate_positions.keys()))
    
    fig, ax = plt.subplots(figsize=(14,8))
    if positions.size > 0:
        ax.scatter(positions[:,0], positions[:,1], c='purple', s=50, edgecolors='k', alpha=0.7,
                   label=f"Duplicates ({len(positions)})")
    ax.set_xlabel("Horizontal Position (µm)")
    ax.set_ylabel("Vertical Position (µm)")
    ax.set_title("Neuron Status Overlay Across Datasets")
    ax.legend()
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, "neuron_status_overlay.png"), bbox_inches='tight')
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
def plot_sttc_layout(sd, sttc_threshold=0.1, xlim=None, ylim=None, output_path=None):
    """
    Create a network-style layout plot based on neuron positions.
    This function computes a filtered STTC matrix (using the "difference" method)
    from the SpikeData object and draws connections between neurons whose filtered STTC
    exceeds the specified threshold.

    Parameters:
        sd : SpikeData
            A SpikeData object with:
              - sd.train: list of spike trains (in ms).
              - sd.length: recording duration (ms).
              - sd.neuron_data: dictionary (or list with dictionary) with neuron 'position'.
        sttc_threshold : float, optional
            Minimum filtered STTC value for a connection to be drawn (default 0.1).
        xlim : tuple, optional
            x-axis limits (µm).
        ylim : tuple, optional
            y-axis limits (µm).
        output_path : str, optional
            Directory to save the plot.

    Returns:
        fig, ax : Matplotlib figure and axes objects.
    """
    # Use the first element if sd.neuron_data is a list.
    data_source = sd.neuron_data[0] if isinstance(sd.neuron_data, list) else sd.neuron_data
    neuron_xy = []
    for neuron in data_source.values():
        pos = neuron.get('position', None)
        if pos is not None:
            neuron_xy.append(pos)
    neuron_xy = np.array(neuron_xy)
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(neuron_xy[:,0], neuron_xy[:,1], alpha=0.15, c='b')
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel("Horizontal Position (µm)")
    ax.set_ylabel("Vertical Position (µm)")
    ax.set_title("STTC Layout Plot")
    
    # Compute the filtered STTC matrix using the "difference" method.
    # (Assuming we want to use the local randomization method.)
    filtered_sttc = compute_filtered_sttc(sd.train, sd.length, delt=20, n_shuffles=10,
                                          filter_method="difference", randomization_method="global")
    N = filtered_sttc.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            if filtered_sttc[i, j] < sttc_threshold:
                continue
            pos_i = data_source.get(i, {}).get('position', None)
            pos_j = data_source.get(j, {}).get('position', None)
            if pos_i is None or pos_j is None:
                continue
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]],
                    linewidth=filtered_sttc[i, j], c='k')
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, "sttc_layout.png"))
        plt.close(fig)
    return fig, ax


def plot_avg_distance_histogram_overlaid(
    sd_list,
    dataset_names,
    bins_range=(0, 1000),
    n_bins=50,
    output_path=None
):
    """
    Compute and plot overlaid average-distance histograms for multiple SpikeData objects
    on a single figure.

    For each SpikeData in sd_list:
      1. Retrieve neuron positions via get_neuron_positions(sd).
      2. Compute the pairwise distance matrix and the average distance per neuron.
      3. Build a histogram (counts vs. distance bin centers).
    Then overlay these histograms on a single plot.

    Parameters:
    -----------
    sd_list : list of SpikeData
        A list of SpikeData objects (e.g., from the same organoid but different days).
    dataset_names : list of str
        Names/labels for each dataset (same length as sd_list).
    bins_range : tuple of (float, float), default (0, 1000)
        The min and max for the x-axis in µm.
    n_bins : int, default 50
        Number of bins for the histogram.
    output_path : str, optional
        If provided, the figure is saved to this path (PNG). Otherwise, plotted interactively.
        e.g., "/path/to/avg_distance_overlay.png"
    figsize : tuple, default (8, 6)
        Figure size in inches.
    """
    # A small color palette for multiple datasets
    colors = ["green", "lime", "royalblue", "cyan"]

    plt.figure(figsize=(10, 8))

    for i, sd in enumerate(sd_list):
        dataset_name = dataset_names[i] if i < len(dataset_names) else f"Dataset_{i+1}"
        c = colors[i % len(colors)]
        
        # 1. Get positions using your get_neuron_positions function.
        full_positions = get_neuron_positions(sd)
        valid_positions = []
        # Filter out invalid [0, 0] positions if needed.
        for pos in full_positions:
            if not np.allclose(pos, [0, 0]):
                valid_positions.append(pos)
        # Convert list to a NumPy array.
        valid_positions = np.array(valid_positions)
        
        if valid_positions.size == 0:
            print(f"No valid positions for dataset {dataset_name}, skipping.")
            continue
        
        # 2. Compute pairwise distances.
        avg_distance = compute_average_distance(valid_positions)

        # 3. Build histogram.
        bin_edges = np.linspace(bins_range[0], bins_range[1], n_bins + 1)
        counts, _ = np.histogram(avg_distance, bins=bin_edges)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Overlay the histogram as a line plot.
        plt.plot(bin_centers, counts, marker='o', linestyle='-', color=c, label=dataset_name)
    
    plt.xlabel("Average Distance to all Neurons(µm)")
    plt.ylabel("Number of Neurons")
    plt.title("Average Distance Per Neuron")
    plt.xlim(bins_range)
    plt.legend()
    plt.tight_layout()

    if output_path:
        # If output_path is a directory, save with a default filename.
        if os.path.isdir(output_path):
            save_path = os.path.join(output_path, "avg_distance_histogram_overlaid.png")
        else:
            save_path = output_path
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()





# -----------------------------------------------------------------------------
# For testing purposes, you can run this module interactively.
if __name__ == '__main__':
    # In a notebook you would import these functions and pass a SpikeData object.
    # For example:
    # from load_npz import load_spikedata
    # sd = load_spikedata("path/to/data.npz")
    # fig1, ax1 = raw_footprint(sd)
    # fig2, ax2 = overlay_fr_footprint(sd)
    # fig3, ax3 = plot_neuron_status([sd.neuron_data], [sd.cleaned_names[0]])
    # fig4, ax4 = plot_sttc_layout(sd, sttc_threshold=0.1, xlim=(0,2000), ylim=(0,2000))
    pass
