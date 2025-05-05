#!/usr/bin/env python3
"""
Module: plot_burst.py

This module provides functions to plot bursting metrics from a SpikeData object.
It assumes that a burst-detection function (e.g. burst_detection) is available 
to extract bursts from each neuron's spike train.

Functions included:
    1. plot_ibi_histogram(sd, output_path=None, neuron_indices=None, time_window=None, 
                           bins=50, log_scale=False, kde=False, burst_threshold, spike_num_thr)
         - Computes inter-burst intervals (IBIs) across neurons and plots a histogram.
    2. plot_inverse_ibi_histogram(...)
         - Same as above but plots the histogram of 1/IBI.
    3. plot_burst_participation_violin(sd, output_path=None, neuron_indices=None, 
                                        burst_threshold, spike_num_thr)
         - Computes the burst count (participation) per neuron and plots a violin plot.
    4. plot_burst_freq_vs_duration(sd, output_path=None, neuron_indices=None, 
                                    burst_threshold, spike_num_thr)
         - For each detected burst, computes its duration and (for its neuron) the burst frequency 
           (number of bursts per second) and plots a scatter plot.
    5. (Optional) plot_rank_order_correlation(sd, ...) [stub]
         - Computes and plots the distribution of Spearman rank-order correlations among neurons’ burst orders.
         
Notes:
  - time_window is given in seconds; spike times in sd.train are assumed to be in ms.
  - burst_threshold and spike_num_thr are parameters for burst detection.
  - If output_path is provided, the plot is saved; otherwise, the (fig, ax) objects are returned.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde, spearmanr

# Import burst_detection from your spikedata module (adjust import path as needed)
from analysis.analysis.bursting_sd import burst_detection

# --- Helper: Compute bursts for each neuron ---
def compute_bursts(sd, burst_threshold, spike_num_thr=3, neuron_indices=None, time_window=None):
    """
    Compute bursts for each neuron using the burst_detection function.
    
    Parameters:
        sd : SpikeData object.
        burst_threshold : float
            The inter-spike interval threshold (ms) for detecting a burst.
        spike_num_thr : int, optional
            Minimum number of spikes to count as a burst.
        neuron_indices : list or array, optional
            If provided, only process these neurons.
        time_window : tuple, optional
            (start, end) in seconds; if provided, restrict spike times to this window.
    
    Returns:
        bursts_list : list of lists
            Each element is a list of bursts for one neuron. Each burst is a tuple:
            (burst_start_time, burst_end_time, spike_count)
    """
    bursts_list = []
    indices = neuron_indices if neuron_indices is not None else range(len(sd.train))
    for i in indices:
        spikes = sd.train[i]
        # Restrict to time window if provided (convert seconds to ms)
        if time_window is not None:
            start_ms, end_ms = np.array(time_window) * 1000.0
            spikes = spikes[(spikes >= start_ms) & (spikes <= end_ms)]
        if spikes.size < 2:
            bursts_list.append([])
            continue
        # burst_detection returns [ [start_index, spike_count], ... ] and burst_set (not used here)
        burst_features, _ = burst_detection(spikes, burst_threshold, spike_num_thr)
        neuron_bursts = []
        for feat in burst_features:
            start_idx, count = feat
            # Compute burst start and end times (in ms)
            burst_start = spikes[start_idx]
            burst_end = spikes[start_idx + count - 1]
            neuron_bursts.append((burst_start, burst_end, count))
        bursts_list.append(neuron_bursts)
    return bursts_list

# --- Function 1: Plot IBI Histogram ---
def plot_ibi_histogram(sd, output_path=None, neuron_indices=None, time_window=None, 
                       bins=50, log_scale=False, kde=False, burst_threshold=50, spike_num_thr=3):
    """
    Compute inter-burst intervals (IBIs) for each neuron and plot a histogram.
    
    Parameters:
        sd : SpikeData object.
        output_path : str, optional.
        neuron_indices : list/array, optional.
        time_window : tuple, optional (start, end in seconds).
        bins : int or array-like.
        log_scale : bool.
        kde : bool.
        burst_threshold : float, in ms, parameter for burst_detection.
        spike_num_thr : int, parameter for burst_detection.
    
    Returns:
        fig, ax.
    """
    bursts_list = compute_bursts(sd, burst_threshold, spike_num_thr, neuron_indices, time_window)
    # Compute IBIs (in seconds) per neuron. We'll compute the difference between consecutive burst start times.
    all_ibi = []
    for bursts in bursts_list:
        if len(bursts) < 2:
            continue
        # Extract burst onset times
        onsets = [b[0] for b in bursts]
        # Compute differences and convert ms->s
        ibi = np.diff(onsets) / 1000.0
        all_ibi.extend(ibi.tolist())
    all_ibi = np.array(all_ibi)
    if all_ibi.size == 0:
        raise ValueError("No IBI data available.")
    
    fig, ax = plt.subplots(figsize=(12,6))
    if log_scale:
        min_val = np.min(all_ibi[all_ibi > 0])
        max_val = np.max(all_ibi)
        bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), bins)
        ax.hist(all_ibi, bins=bin_edges, color='red', alpha=0.7, label="IBI")
        ax.set_xscale('log')
    else:
        ax.hist(all_ibi, bins=bins, color='red', alpha=0.7, label="IBI")
    if kde:
        density = gaussian_kde(all_ibi)
        xs = np.linspace(np.min(all_ibi), np.max(all_ibi), 500)
        ax.plot(xs, density(xs), color='blue', linewidth=2, label="KDE")
    ax.set_xlabel("Inter-Burst Interval (s)")
    ax.set_ylabel("Count")
    ax.legend()
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "ibi_histogram.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# --- Function 2: Plot Inverse IBI Histogram ---
def plot_inverse_ibi_histogram(sd, output_path=None, neuron_indices=None, time_window=None, 
                               bins=50, log_scale=False, kde=False, burst_threshold=50, spike_num_thr=3):
    """
    Compute the inverse inter-burst intervals (1/IBI) and plot a histogram.
    
    Parameters are the same as for plot_ibi_histogram.
    
    Returns:
        fig, ax.
    """
    bursts_list = compute_bursts(sd, burst_threshold, spike_num_thr, neuron_indices, time_window)
    all_ibi = []
    for bursts in bursts_list:
        if len(bursts) < 2:
            continue
        onsets = [b[0] for b in bursts]
        ibi = np.diff(onsets) / 1000.0
        all_ibi.extend(ibi.tolist())
    all_ibi = np.array(all_ibi)
    if all_ibi.size == 0:
        raise ValueError("No IBI data available.")
    inv_ibi = 1.0 / all_ibi
    # Remove extreme values if needed.
    inv_ibi = inv_ibi[np.isfinite(inv_ibi)]
    
    fig, ax = plt.subplots(figsize=(12,6))
    if log_scale:
        min_val = np.min(inv_ibi[inv_ibi > 0])
        max_val = np.max(inv_ibi)
        bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), bins)
        ax.hist(inv_ibi, bins=bin_edges, color='purple', alpha=0.7, label="1/IBI")
        ax.set_xscale('log')
    else:
        ax.hist(inv_ibi, bins=bins, color='purple', alpha=0.7, label="1/IBI")
    if kde:
        density = gaussian_kde(inv_ibi)
        xs = np.linspace(np.min(inv_ibi), np.max(inv_ibi), 500)
        ax.plot(xs, density(xs), color='darkmagenta', linewidth=2, label="KDE")
    ax.set_xlabel("Inverse Inter-Burst Interval (Hz)")
    ax.set_ylabel("Count")
    ax.legend()
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "inverse_ibi_histogram.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# --- Function 3: Plot Burst Participation Violin ---
def plot_burst_participation_violin(sd, output_path=None, neuron_indices=None, burst_threshold=50, spike_num_thr=3):
    """
    Compute the burst participation (number of bursts per neuron) and plot a violin plot.
    
    Parameters:
        sd : SpikeData object.
        output_path : str, optional.
        neuron_indices : list, optional.
        burst_threshold : float, ms.
        spike_num_thr : int.
    
    Returns:
        fig, ax.
    """
    bursts_list = compute_bursts(sd, burst_threshold, spike_num_thr, neuron_indices)
    # Compute burst counts per neuron.
    burst_counts = np.array([len(bursts) for bursts in bursts_list])
    fig, ax = plt.subplots(figsize=(8,6))
    ax.violinplot(burst_counts, showmeans=True)
    ax.set_ylabel("Number of Bursts")
    ax.set_title("Burst Participation Across Neurons")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "burst_participation_violin.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# --- Function 4: Plot Burst Frequency vs Duration ---
def plot_burst_freq_vs_duration(sd, output_path=None, neuron_indices=None, burst_threshold=50, spike_num_thr=3):
    """
    For each burst detected in the SpikeData object, compute its duration and the burst frequency 
    (defined as the number of bursts per second for that neuron) and plot them as a scatter plot.
    
    Parameters:
        sd : SpikeData object.
        output_path : str, optional.
        neuron_indices : list, optional.
        burst_threshold : float, ms.
        spike_num_thr : int.
    
    Returns:
        fig, ax.
    """
    bursts_list = compute_bursts(sd, burst_threshold, spike_num_thr, neuron_indices)
    burst_durations = []
    burst_frequencies = []
    # Loop over neurons.
    for i, bursts in enumerate(bursts_list):
        # If no bursts, skip
        if len(bursts) == 0:
            continue
        # Compute burst durations (in seconds) for each burst in the neuron.
        durations = [(burst[1] - burst[0]) / 1000.0 for burst in bursts]
        burst_durations.extend(durations)
        # Burst frequency for this neuron: number of bursts divided by recording duration (in seconds)
        freq = len(bursts) / (sd.length / 1000.0)
        burst_frequencies.extend([freq] * len(durations))
    
    if len(burst_durations) == 0:
        raise ValueError("No burst data available for frequency vs. duration plot.")
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(burst_frequencies, burst_durations, alpha=0.7, color='darkblue')
    ax.set_xlabel("Burst Frequency (Hz)")
    ax.set_ylabel("Burst Duration (s)")
    ax.set_title("Burst Frequency vs. Burst Duration")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "burst_freq_vs_duration.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# --- Function 5: Plot Rank Order Correlation (stub) ---
def plot_rank_order_correlation(sd, output_path=None, neuron_indices=None, burst_threshold=50, spike_num_thr=3):
    """
    (Stub) Compute the rank order correlation (Spearman) between the burst onset orders of neurons.
    
    Parameters:
        sd : SpikeData object.
        output_path : str, optional.
        neuron_indices : list, optional.
        burst_threshold : float, ms.
        spike_num_thr : int.
    
    Returns:
        fig, ax.
    """
    # This is a placeholder. One approach is:
    #   For each neuron, get a vector of burst onset times.
    #   Rank order these vectors (or the order in which neurons burst)
    #   Compute pairwise Spearman correlations and plot a histogram of these values.
    indices = neuron_indices if neuron_indices is not None else range(len(sd.train))
    burst_onsets = []
    bursts_list = compute_bursts(sd, burst_threshold, spike_num_thr, neuron_indices)
    for bursts in bursts_list:
        if bursts:
            # Use the first spike time of each burst
            burst_onsets.append([b[0] for b in bursts])
        else:
            burst_onsets.append([])
    
    # For simplicity, compute pairwise Spearman correlations between neurons that have bursts.
    correlations = []
    for i in range(len(burst_onsets)):
        for j in range(i+1, len(burst_onsets)):
            if len(burst_onsets[i]) > 0 and len(burst_onsets[j]) > 0:
                # Use Spearman correlation on the burst onset lists.
                # Note: In practice, you might need to align the bursts in time or use a different method.
                rho, _ = spearmanr(burst_onsets[i], burst_onsets[j])
                correlations.append(rho)
    correlations = np.array(correlations)
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.hist(correlations, bins=50, color='teal', alpha=0.7)
    ax.set_xlabel("Spearman Rank-Order Correlation")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Rank Order Correlations Among Neurons")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "rank_order_correlation.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# -----------------------------------------------------------------------------
# For testing purposes:
if __name__ == '__main__':
    # Example usage (assuming you have a load function to get a SpikeData object):
    # from load_data.load_npz import load_spikedata
    # sd = load_spikedata("path/to/data.npz")
    # fig1, ax1 = plot_ibi_histogram(sd, output_path="output/", burst_threshold=50, spike_num_thr=3)
    # fig2, ax2 = plot_inverse_ibi_histogram(sd, output_path="output/", burst_threshold=50, spike_num_thr=3)
    # fig3, ax3 = plot_burst_participation_violin(sd, output_path="output/", burst_threshold=50, spike_num_thr=3)
    # fig4, ax4 = plot_burst_freq_vs_duration(sd, output_path="output/", burst_threshold=50, spike_num_thr=3)
    # fig5, ax5 = plot_rank_order_correlation(sd, output_path="output/", burst_threshold=50, spike_num_thr=3)
    pass
