#!/usr/bin/env python3
"""
Module: plot_rasters.py

This module provides functions for plotting raster plots from a SpikeData object.
It includes:
    1. plot_raster: Basic spike raster plot with dual x-axes (seconds and minutes)
       and a simplified y-axis labeling (only 5 tick marks).
    2. plot_high_activity_rasters: Plots raster plots over short time windows
       centered around high-activity periods (based on the population firing rate).
    3. overlay_fr_raster: Plots a raster with an overlay of the population firing rate curve.

Each function returns the matplotlib figure and axes. If an output_path is provided,
the figure is saved there; otherwise, it can be displayed interactively.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# For secondary axis functionality
from matplotlib.ticker import MaxNLocator

# We assume the SpikeData object has:
# - a list attribute "train" where each element is an array of spike times (in ms)
# - a "durations" list (one per dataset) with the recording duration (in ms)
# - a "N" attribute (number of neurons)
# - a "raster" method that bins spikes given a bin_size (in ms)
# - a "rates" method that returns firing rates (we already have population_firing_rate defined elsewhere)
# - "cleaned_names" (list of dataset names)
# - "firing_rates_list" (list of per-dataset firing rates)
# We'll assume "sd" is a SpikeData object.

def _set_dual_xaxis(ax, total_time_s):
    """
    Set a secondary x-axis on the provided axes.
    Primary axis: time in seconds.
    Secondary axis: time in minutes.
    """
    secax = ax.secondary_xaxis('top', functions=(lambda s: s/60, lambda m: m*60))
    secax.set_xlabel("Time (min)")
    return ax

def _set_custom_y_ticks(ax, n_neurons):
    """
    Set y-axis ticks so that only 5 tick marks (including first and last)
    are shown to avoid overcrowding.
    """
    tick_positions = np.linspace(0, n_neurons - 1, num=5, dtype=int)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_positions)
    ax.set_ylabel("Neuron Index")
    return ax

def plot_raster(sd, neuron_indices=None, output_path=None):
    """
    Plot a basic spike raster.

    Parameters:
        sd : SpikeData
            A SpikeData object.
        neuron_indices : list or np.ndarray, optional
            Indices of neurons to plot. If None, all neurons are plotted.
        output_path : str, optional
            If provided, the figure is saved to this path; otherwise, the figure is returned.

    Returns:
        fig, ax : matplotlib figure and axes objects.
    """
    # If neuron_indices is not provided, use all neurons
    if neuron_indices is None:
        neuron_indices = np.arange(len(sd.train))
    else:
        neuron_indices = np.array(neuron_indices)

    n_neurons = len(neuron_indices)
    # Create figure and axes.
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each neuron's spikes. Convert spike times from ms to seconds.
    for idx, neuron_idx in enumerate(neuron_indices):
        spikes = sd.train[neuron_idx]
        if spikes.size > 0:
            # Convert spike times from ms to seconds.
            ax.scatter(spikes / 1000.0, [idx] * len(spikes), marker="|", c='k', s=4, alpha=0.7)
    
    # Set x-axis limits (in seconds).
    total_time_s = sd.length / 1000.0
    ax.set_xlim(0, total_time_s)
    ax.set_xlabel("Time (s)")
    # Create secondary x-axis: time in minutes.
    _set_dual_xaxis(ax, total_time_s)
    
    # Set custom y-axis ticks.
    _set_custom_y_ticks(ax, n_neurons)
    
    # Optionally save or show.
    if output_path:
        # Ensure output directory exists.
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "raster_plot.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

def plot_high_activity_rasters(sd, output_path=None, window_sizes=[60, 30, 10], activity_percentile=95):
    """
    Plot raster plots around high-activity periods.
    
    For each dataset in sd, compute the population firing rate (using 1ms bins),
    then identify time points where the firing rate exceeds the given percentile.
    For each such time point and for each specified window size (in seconds),
    a raster plot is generated showing spikes within that time window.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        output_path : str, optional
            Directory where plots will be saved (if provided). If None, figures are returned.
        window_sizes : list of float, optional
            List of window sizes (in seconds) for which to generate raster plots.
        activity_percentile : float, optional
            Percentile of population firing rate used as threshold (default 95).
    
    Returns:
        figs : list of matplotlib figure objects
        axes : list of corresponding axes objects
    """
    # Compute population firing rate from the raster.
    # We use the sd.raster method to create a dense raster with 1ms bins.
    raster = sd.raster(bin_size=1)  # raster: shape (N, T), where T is in ms
    # Sum across neurons to get population spike count per ms.
    pop_spike_counts = np.array(raster.sum(axis=0))
    # Smooth population activity with a moving average (optional) - here we use a simple smoothing over 100ms.
    from scipy.ndimage import uniform_filter1d
    pop_rate = uniform_filter1d(pop_spike_counts, size=100)
    # Convert to Hz per neuron (spikes/ms * 1000 / N)
    pop_rate = pop_rate * 1000.0 / sd.N
    # Create time array in seconds (each bin is 1ms)
    T = raster.shape[1]
    time_array = np.arange(T) / 1000.0

    # Determine threshold from the specified percentile.
    threshold = np.percentile(pop_rate, activity_percentile)
    high_activity_times = time_array[pop_rate > threshold]
    if high_activity_times.size == 0:
        print("No high-activity periods found.")
        return [], []
    
    figs = []
    axes = []
    # For each window size, generate a raster plot around each high activity time.
    for window_size in window_sizes:
        half_window = window_size / 2.0
        for t_center in high_activity_times:
            start_t = max(0, t_center - half_window)
            end_t = start_t + window_size
            # Create a new figure for this window.
            fig, ax = plt.subplots(figsize=(12, 8))
            # Plot spikes for all neurons (convert spike times to seconds)
            for idx, spikes in enumerate(sd.train):
                # Only plot spikes within the window.
                spikes_in_window = spikes[ (spikes/1000.0 >= start_t) & (spikes/1000.0 < end_t) ]
                if spikes_in_window.size > 0:
                    ax.scatter(spikes_in_window/1000.0, [idx]*len(spikes_in_window), marker="|", c='k', s=4, alpha=0.7)
            ax.set_xlim(start_t, end_t)
            ax.set_xlabel("Time (s)")
            _set_dual_xaxis(ax, end_t - start_t)
            _set_custom_y_ticks(ax, len(sd.train))
            ax.set_title(f"High-Activity Raster (Window {window_size}s, Center ~{t_center:.2f}s)")
            figs.append(fig)
            axes.append(ax)
            if output_path:
                os.makedirs(os.path.join(output_path, f"high_activity_{window_size}s"), exist_ok=True)
                fname = os.path.join(output_path, f"high_activity_{window_size}s", f"raster_{int(t_center)}s.png")
                plt.savefig(fname)
                plt.close(fig)
    return figs, axes

def overlay_fr_raster(sd, neuron_indices=None, output_path=None):
    """
    Plot a raster with the population firing rate overlaid.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        neuron_indices : list or np.ndarray, optional
            List of neuron indices to include in the raster. If None, all neurons are used.
        output_path : str, optional
            If provided, the plot is saved to this directory.
    
    Returns:
        fig, ax : matplotlib figure and axes objects.
    """
    if neuron_indices is None:
        neuron_indices = np.arange(len(sd.train))
    else:
        neuron_indices = np.array(neuron_indices)
    n_neurons = len(neuron_indices)
    # Create figure and axis.
    fig, ax = plt.subplots(figsize=(16, 6))
    # Plot raster for each neuron (spike times in seconds)
    for idx, neuron_idx in enumerate(neuron_indices):
        spikes = sd.train[neuron_idx]
        if spikes.size > 0:
            ax.scatter(spikes/1000.0, [idx]*len(spikes), marker="|", c='k', s=4, alpha=0.7)
    
    # Get population firing rate using the provided function.
    # We assume sd.raster() creates a dense raster with bin size in ms.
    # Here, we choose a bin_size of 20 ms.
    raster = sd.raster(bin_size=20)
    spikes_per_bin = np.array(raster.sum(axis=0))
    fr = spikes_per_bin / sd.N / (20/1000.0)  # Hz per neuron
    # Create bin centers in seconds
    num_bins = raster.shape[1]
    time_bins = np.linspace(0, sd.length/1000.0, num_bins, endpoint=False) + 20/2000.0
    
    # Overlay the firing rate curve on a secondary axis.
    ax2 = ax.twinx()
    ax2.plot(time_bins, fr, color='r', linewidth=2, alpha=0.7)
    ax2.set_ylabel("Population Firing Rate (Hz/neuron)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Set x-axis for time in seconds.
    ax.set_xlim(0, sd.length/1000.0)
    ax.set_xlabel("Time (s)")
    _set_dual_xaxis(ax, sd.length/1000.0)
    _set_custom_y_ticks(ax, n_neurons)
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "overlay_fr_raster.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# For testing purposes, you can run this module interactively:
if __name__ == '__main__':
    # For example usage, you would load a SpikeData object (sd) and then call:
    # fig, ax = plot_raster(sd)
    # figs, axes = plot_high_activity_rasters(sd)
    # fig2, ax2 = overlay_fr_raster(sd)
    # In a Jupyter notebook, simply returning fig, ax will allow the cell to display the plot.
    pass
