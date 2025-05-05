#!/usr/bin/env python3
"""
Module: plot_isi.py

This module provides functions to plot inter-spike interval (ISI) statistics
from a SpikeData object.

Functions:
    1. plot_isi_histogram(sd, output_path=None, neuron_indices=None, time_window=None, 
                           bins=50, log_scale=False, kde=False)
         - Computes ISIs across neurons (or a subset) and plots a histogram.
           If log_scale is True, logarithmically spaced bins are used.
           If kde is True, overlays a kernel density estimate.
    2. plot_cv_of_isi(sd, output_path=None, neuron_indices=None, bins=50, xlim=None)
         - Computes the CV (coefficient of variation) of ISIs for each neuron and plots a histogram.
         
Note:
    - time_window is given as a tuple (start, end) in seconds.
    - neuron_indices (or for single neuron plots, neuron_number) is optional.
    - If output_path is provided, the figure is saved and closed; otherwise, the figure and axes are returned.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
def _compute_all_isis(sd, neuron_indices=None, time_window=None):
    """
    Compute ISIs for a subset of neurons (or all neurons if None).
    Optionally restrict to a time window (in seconds).
    
    Returns:
        all_isis : list
            List of ISI values (in seconds) across the specified neurons.
    """
    all_isis = []
    # If neuron_indices not provided, use all neurons.
    indices = neuron_indices if neuron_indices is not None else range(len(sd.train))
    
    # Loop over each specified neuron.
    for i in indices:
        spikes = sd.train[i]
        # spikes are in ms; if a time window is provided (in seconds), convert it to ms.
        if time_window is not None:
            start_ms, end_ms = np.array(time_window) * 1000.0
            # Keep spikes within the window.
            spikes = spikes[(spikes >= start_ms) & (spikes <= end_ms)]
        if spikes.size < 2:
            continue
        isis = np.diff(spikes) / 1000.0  # convert ISI to seconds
        all_isis.extend(isis.tolist())
    return np.array(all_isis)

# ---------------------------------------------------------------------------
def plot_isi_histogram(sd, output_path=None, neuron_indices=None, time_window=None,
                         bins=50, log_scale=False, kde=False):
    """
    Plot a histogram of inter-spike intervals (ISIs) from a SpikeData object.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        output_path : str, optional
            Directory to save the plot. If None, returns (fig, ax).
        neuron_indices : array-like, optional
            Indices of neurons to include. If None, uses all neurons.
        time_window : tuple, optional
            (start, end) in seconds. If provided, only ISIs from spikes within that time window are used.
        bins : int or array-like, optional
            Number of bins or bin edges for the histogram.
        log_scale : bool, optional
            If True, use logarithmically spaced bins.
        kde : bool, optional
            If True, overlay a kernel density estimate.
    
    Returns:
        fig, ax : matplotlib figure and axes objects.
    """
    all_isis = _compute_all_isis(sd, neuron_indices, time_window)
    if all_isis.size == 0:
        raise ValueError("No ISI data available for the specified neurons/time window.")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    if log_scale:
        # Create logarithmic bins from min to max.
        min_val = np.min(all_isis[all_isis > 0])
        max_val = np.max(all_isis)
        bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), bins)
        ax.hist(all_isis, bins=bin_edges, color='red', alpha=0.7, label='Histogram')
        ax.set_xscale('log')
    else:
        ax.hist(all_isis, bins=bins, color='red', alpha=0.7, label='Histogram')
    if kde:
        density = gaussian_kde(all_isis)
        xs = np.linspace(np.min(all_isis), np.max(all_isis), 500)
        ax.plot(xs, density(xs), color='blue', linewidth=2, label='KDE')
    ax.set_xlabel("Inter-Spike Interval (s)")
    ax.set_ylabel("Count")
    ax.legend()
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "isi_histogram.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
def plot_cv_of_isi(sd, output_path=None, neuron_indices=None, bins=50, xlim=None):
    """
    Plot the histogram of the coefficient of variation (CV) of ISIs for each neuron.
    
    CV is defined as the standard deviation divided by the mean of the ISIs.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        output_path : str, optional
            Directory to save the plot.
        neuron_indices : array-like, optional
            Neuron indices to include. If None, uses all neurons.
        bins : int, optional
            Number of bins for the histogram.
        xlim : tuple, optional
            (min, max) limits for the x-axis.
    
    Returns:
        fig, ax : matplotlib figure and axes objects.
    """
    # If neuron_indices not provided, use all neurons.
    indices = neuron_indices if neuron_indices is not None else range(len(sd.train))
    cv_values = []
    for i in indices:
        spikes = sd.train[i]
        if spikes.size < 2:
            continue
        isis = np.diff(spikes) / 1000.0  # convert to seconds
        if isis.size < 2:
            cv_values.append(np.nan)
        else:
            cv = np.std(isis) / np.mean(isis)
            cv_values.append(cv)
    cv_values = np.array(cv_values)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(cv_values, bins=bins, color='orange', alpha=0.7)
    ax.set_xlabel("CV of ISIs")
    ax.set_ylabel("Count")
    if xlim is not None:
        ax.set_xlim(xlim)
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "cv_of_isi.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # For testing, load a SpikeData object (for example, via your load_npz.py module)
    # and call:
    # fig1, ax1 = plot_isi_histogram(sd, time_window=(0, 10), log_scale=True, kde=True)
    # fig2, ax2 = plot_cv_of_isi(sd)
    pass
