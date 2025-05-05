#!/usr/bin/env python3
"""
Module: plot_fr.py

This module contains functions to generate firing rate plots from a SpikeData object.
Each function accepts an optional time_window (in seconds) and optional neuron selection.
If no time window is provided, the full recording is used.
If no neuron selection is provided, all neurons are used.

Functions include:
    1. plot_smoothed_population_fr:
         Plot the smoothed population firing rate over time.
         Optional parameters:
             - time_window: tuple (start, end) in seconds.
             - neuron_indices: an array of neuron indices to include (default: all).
    2. plot_firing_rate_histogram:
         Plot a histogram of the average firing rates across neurons.
         Optional parameter: neuron_indices.
    3. plot_firing_rate_cdf:
         Plot the cumulative distribution function (CDF) of average firing rates.
         Optional parameter: neuron_indices.
    4. plot_instantaneous_firing_rate:
         Compute and plot the instantaneous firing rate as a line plot or histogram.
         Optional parameter: neuron_number (if None, plots for all neurons).

All functions return (fig, ax) for interactive use if output_path is not provided.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from analysis.firing_sd import population_firing_rate

# ---------------------------------------------------------------------------
def plot_smoothed_population_fr(sd, output_path=None, time_window=None, bin_size=20, sigma=5, neuron_indices=None):
    """
    Plot the smoothed population firing rate over time.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        output_path : str, optional
            Directory to save the plot. If None, the figure is returned.
        time_window : tuple, optional
            (start, end) in seconds; if provided, only data within this window is plotted.
        bin_size : float, optional
            Bin size in ms (default 20 ms).
        sigma : float, optional
            Sigma for Gaussian smoothing (default 5).
        neuron_indices : array-like, optional
            Neuron indices to include (default: all).
    
    Returns:
        fig, ax : matplotlib figure and axes objects.
    """
    time_bins, fr = population_firing_rate(sd, bin_size, sigma, neuron_indices)
    # Convert time bins from ms to seconds.
    time_bins_s = time_bins / 1000.0
    if time_window is not None:
        start, end = time_window
        mask = (time_bins_s >= start) & (time_bins_s <= end)
        time_bins_s = time_bins_s[mask]
        fr = fr[mask]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_bins_s, fr, color='r', linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population Firing Rate (Hz/neuron)")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "smoothed_population_fr.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
def plot_firing_rate_histogram(sd, output_path=None, neuron_indices=None):
    """
    Plot a histogram of the average firing rates across neurons.
    
    Parameters:
        sd : SpikeData
            A SpikeData object. Firing rates are obtained via sd.rates(unit="Hz")
            or from sd.firing_rates_list[0].
        output_path : str, optional
            Directory to save the plot.
        neuron_indices : array-like, optional
            Neuron indices to include. If None, uses all neurons.
    
    Returns:
        fig, ax : matplotlib figure and axes objects.
    """
    try:
        firing_rates = sd.rates(unit="Hz")
    except AttributeError:
        firing_rates = sd.firing_rates_list[0]
    firing_rates = np.array(firing_rates)
    if neuron_indices is not None:
        firing_rates = firing_rates[np.array(neuron_indices)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(firing_rates, bins=50, color='green', alpha=0.7)
    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("Count")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "firing_rate_histogram.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
def plot_firing_rate_cdf(sd, output_path=None, neuron_indices=None):
    """
    Plot the cumulative distribution function (CDF) of the average firing rates.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        output_path : str, optional
            Directory to save the plot.
        neuron_indices : array-like, optional
            Neuron indices to include.
    
    Returns:
        fig, ax : matplotlib figure and axes objects.
    """
    try:
        firing_rates = sd.rates(unit="Hz")
    except AttributeError:
        firing_rates = sd.firing_rates_list[0]
    firing_rates = np.array(firing_rates)
    if neuron_indices is not None:
        firing_rates = firing_rates[np.array(neuron_indices)]
    
    sorted_rates = np.sort(firing_rates)
    cdf = np.arange(len(sorted_rates)) / float(len(sorted_rates))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sorted_rates, cdf, color='purple')
    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("CDF")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fname = os.path.join(output_path, "firing_rate_cdf.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
def instantaneous_firing_rate(sd, neuron_number):
    """
    Compute the instantaneous firing rate for a single neuron based on its inter-spike intervals.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        neuron_number : int
            The index of the neuron to analyze.
    
    Returns:
        inst_fr : np.ndarray
            An array of instantaneous firing rate values (Hz) computed as the inverse of ISI.
    """
    # Round spike times to the nearest ms and remove duplicates.
    spikes = np.unique(np.round(sd.train[neuron_number]).astype(int))
    if spikes.size < 2:
        return np.array([])
    # Compute ISIs (in ms) and instantaneous rate = 1/ISI (converted to Hz)
    isis = np.diff(spikes)
    # Avoid division by zero.
    isis = np.where(isis == 0, 1, isis)
    inst_fr = 1000.0 / isis  # since ISI is in ms, 1000/ISI gives Hz
    return inst_fr

def plot_instantaneous_firing_rate(sd, output_path=None, neuron_number=None, time_window=None, plot_type="line"):
    """
    Plot the instantaneous firing rate.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        output_path : str, optional
            Directory to save the plot.
        neuron_number : int or None, optional
            If an integer, plots for that neuron; if None, plots for all neurons (overlayed).
        time_window : tuple, optional
            (start, end) in seconds; if provided, restricts the plot to that window.
        plot_type : str, optional
            "line" to plot as a time series; "histogram" to plot a histogram of instantaneous rates.
    
    Returns:
        fig, ax : matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if neuron_number is None:
        # Plot instantaneous firing rate for each neuron.
        all_inst_rates = []
        for idx in range(sd.N):
            inst_fr = instantaneous_firing_rate(sd, idx)
            if inst_fr.size == 0:
                continue
            # Create a time axis for this neuron's instantaneous rate.
            # Here we assume each ISI contributes one value and the time axis is cumulative.
            spike_times = np.unique(np.round(sd.train[idx]).astype(int))
            time_axis = np.cumsum(np.insert(np.diff(spike_times), 0, spike_times[0])) / 1000.0
            if time_window is not None:
                start, end = time_window
                mask = (time_axis >= start) & (time_axis <= end)
                time_axis = time_axis[mask]
                inst_fr = inst_fr[mask]
            ax.plot(time_axis, gaussian_filter1d(inst_fr, sigma=5), alpha=0.5, label=f"Neuron {idx}")
            all_inst_rates.extend(inst_fr)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Instantaneous Firing Rate (Hz)")
        ax.legend(fontsize=8)
    else:
        # Plot for a single neuron.
        inst_fr = instantaneous_firing_rate(sd, neuron_number)
        if inst_fr.size == 0:
            ax.text(0.5, 0.5, f"No data for neuron {neuron_number}", ha="center", va="center")
        else:
            # Reconstruct a rough time axis from spike times.
            spike_times = np.unique(np.round(sd.train[neuron_number]).astype(int))
            time_axis = np.cumsum(np.insert(np.diff(spike_times), 0, spike_times[0])) / 1000.0
            if time_window is not None:
                start, end = time_window
                mask = (time_axis >= start) & (time_axis <= end)
                time_axis = time_axis[mask]
                inst_fr = inst_fr[mask]
            if plot_type == "line":
                ax.plot(time_axis, gaussian_filter1d(inst_fr, sigma=5), color='blue', linewidth=2)
                ax.set_ylabel("Instantaneous Firing Rate (Hz)")
            elif plot_type == "histogram":
                ax.hist(inst_fr, bins=50, color='blue', alpha=0.7)
                ax.set_xlabel("Instantaneous Firing Rate (Hz)")
                ax.set_ylabel("Count")
            else:
                raise ValueError("Invalid plot_type. Choose 'line' or 'histogram'.")
            ax.set_xlabel("Time (s)")
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        if neuron_number is None:
            fname = os.path.join(output_path, "instantaneous_firing_rate_all.png")
        else:
            fname = os.path.join(output_path, f"instantaneous_firing_rate_neuron{neuron_number}_{plot_type}.png")
        plt.savefig(fname)
        plt.close(fig)
    return fig, ax

# ---------------------------------------------------------------------------
# For testing purposes, you might run this module interactively:
if __name__ == '__main__':
    # Example (when using in a notebook):
    # from load_npz import load_spikedata
    # sd = load_spikedata("path/to/your/data.zip")
    # fig1, ax1 = plot_smoothed_population_fr(sd, time_window=(0, 10))
    # fig2, ax2 = plot_firing_rate_histogram(sd)
    # fig3, ax3 = plot_firing_rate_cdf(sd)
    # fig4, ax4 = plot_instantaneous_firing_rate(sd, neuron_number=0, plot_type="line", time_window=(0,5))
    # fig5, ax5 = plot_instantaneous_firing_rate(sd, neuron_number=0, plot_type="histogram")
    pass
