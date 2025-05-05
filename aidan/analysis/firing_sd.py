#!/usr/bin/env python3
"""
Module: firing_sd.py

This module provides analysis functions for a SpikeData object.
It includes functions for computing:
    - Mean and instantaneous firing rates
    - Interspike interval (ISI) statistics, including CV and CV2
    - Fano factor per neuron
    - Population firing rate (binned and smoothed)
    - A synchrony index over short time windows
    - A generic randomized comparison helper to filter out metrics that are below the randomized baseline.
    
Functions assume spike times in the SpikeData object are in milliseconds.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from braingeneers.analysis.analysis import SpikeData  

##############################################
# 1. Firing Rate Functions
##############################################

def mean_firing_rates(sd: SpikeData) -> np.ndarray:
    """
    Compute the mean firing rate (Hz) of each neuron.
    
    Parameters:
        sd (SpikeData): SpikeData object.
    
    Returns:
        np.ndarray: Array of mean firing rates (Hz) per neuron.
    """
    mean_rates = []
    # Convert total duration from ms to s.
    duration_sec = sd.length / 1000.0
    for neuron_spikes in sd.train:
        rate = len(neuron_spikes) / duration_sec
        mean_rates.append(rate)
    return np.array(mean_rates)


def instant_firing_rate(sd: SpikeData, neuron_num: int, max_ifr: float = 9e10) -> list:
    """
    Calculate the instantaneous firing rate (Hz) for a single neuron
    as the inverse of the interspike intervals.
    
    Parameters:
        sd (SpikeData): SpikeData object.
        neuron_num (int): Index of the neuron.
        max_ifr (float): Maximum possible instantaneous firing rate to cap outliers.
        
    Returns:
        list: Instantaneous firing rate sampled at every ms (Hz).
    """
    # Round spike times (ms) and remove duplicates
    spike_times = np.unique(np.round(sd.train[neuron_num]).astype(int))
    inst_rate = []
    last_spike = 0
    for spike in spike_times:
        isi = spike - last_spike
        # For each ms in the interval, assign the inverse ISI (capped at max_ifr)
        rate = (1/isi) if isi > 0 else max_ifr
        rate = min(rate, max_ifr)
        inst_rate.extend([rate] * isi)
        last_spike = spike
    return inst_rate


##############################################
# 2. ISI, CV, and CV2 Metrics
##############################################

def isi(sd: SpikeData, neuron: int = None, max_isi: float = None) -> list:
    """
    Compute interspike intervals (ISIs) for one neuron or for all neurons.
    
    Parameters:
        sd (SpikeData): SpikeData object.
        neuron (int, optional): If provided, compute ISI for that neuron only.
                                Otherwise, compute ISI for all neurons.
        max_isi (float, optional): If provided, only return ISIs less than max_isi.
    
    Returns:
        list: List of ISIs (ms). If neuron is None, returns a combined list.
    """
    if neuron is None:
        all_isis = []
        for ns in sd.interspike_intervals():
            if max_isi is not None:
                ns = ns[ns < max_isi]
            all_isis.extend(ns.tolist())
        return all_isis
    else:
        ns = sd.interspike_intervals()[neuron]
        if max_isi is not None:
            ns = ns[ns < max_isi]
        return ns.tolist()


def coefficient_of_variation(sd: SpikeData, neuron: int = None) -> np.ndarray:
    """
    Compute the coefficient of variation (CV) of the ISI for each neuron,
    or for a specific neuron if provided.
    
    Parameters:
        sd (SpikeData): SpikeData object.
        neuron (int, optional): Specific neuron index. If None, compute for all.
        
    Returns:
        np.ndarray: CV value(s). If a neuron is specified, a single float is returned.
    """
    if neuron is None:
        cvs = []
        for ns in sd.interspike_intervals():
            if len(ns) > 1:
                cvs.append(np.std(ns) / np.mean(ns))
            else:
                cvs.append(np.nan)
        return np.array(cvs)
    else:
        ns = sd.interspike_intervals()[neuron]
        if len(ns) > 1:
            return np.std(ns) / np.mean(ns)
        else:
            return np.nan


def cv2(sd: SpikeData, neuron: int = None) -> np.ndarray:
    """
    Compute the CV2 metric for the ISI of each neuron or a specific neuron.
    CV2 is defined as: 2 * |ISI_n+1 - ISI_n| / (ISI_n+1 + ISI_n)
    
    Parameters:
        sd (SpikeData): SpikeData object.
        neuron (int, optional): Specific neuron index. If None, compute for all neurons.
        
    Returns:
        np.ndarray: CV2 value(s). If a neuron is specified, a single float is returned.
    """
    def compute_cv2_for_array(isis):
        if len(isis) < 2:
            return np.nan
        diffs = 2 * np.abs(np.diff(isis))
        sums = isis[:-1] + isis[1:]
        cv2_vals = diffs / sums
        return np.nanmean(cv2_vals)
    
    if neuron is None:
        cv2_list = []
        isi_list = sd.interspike_intervals()
        for ns in isi_list:
            cv2_list.append(compute_cv2_for_array(ns))
        return np.array(cv2_list)
    else:
        sd_subset = sd.subset(units=neuron)
        isi = sd_subset.interspike_intervals()
        return compute_cv2_for_array(isi)


##############################################
# 3. Fano Factor
##############################################

def fano_factor(sd: SpikeData, bin_size: float = 20) -> np.ndarray:
    """
    Compute the Fano factor (variance/mean) for each neuron's binned spike counts.
    
    Parameters:
        sd (SpikeData): SpikeData object.
        bin_size (float): Bin size (ms) for creating the spike raster.
        
    Returns:
        np.ndarray: Array of Fano factors per neuron.
    """
    # Create a dense raster using the provided bin size.
    # Note: SpikeData.raster returns a dense numpy array.
    raster = sd.raster(bin_size=bin_size)
    fano_factors = []
    for row in raster:
        mean_rate = np.mean(row)
        var_rate = np.var(row)
        # Avoid division by zero by setting mean==0 to 1 (Fano factor = 1 in limit)
        if mean_rate == 0:
            fano_factors.append(1)
        else:
            fano_factors.append(var_rate / mean_rate)
    return np.array(fano_factors)


##############################################
# 4. Population Firing Rate
##############################################

def population_firing_rate(sd: SpikeData, bin_size: float = 20, sigma: float = 5) -> (np.ndarray, np.ndarray):
    """
    Compute the population firing rate (Hz per neuron) over time.
    Spike times are binned, normalized by the number of neurons, and then smoothed with a Gaussian filter.
    
    Parameters:
        sd (SpikeData): SpikeData object.
        bin_size (float): Time bin size (ms).
        sigma (float): Sigma for Gaussian smoothing (in bins).
    
    Returns:
        tuple: (time_bins, firing_rate), where:
            - time_bins: array of time bin centers (ms)
            - firing_rate: smoothed firing rate (Hz per neuron)
    """
    # Create a dense raster with the given bin size.
    raster = sd.raster(bin_size=bin_size)
    # Sum across neurons gives total spikes per bin.
    spikes_per_bin = np.array(raster.sum(axis=0))
    # Normalize by the number of neurons and by bin duration (ms->s conversion)
    firing_rate = spikes_per_bin / sd.N / (bin_size / 1000.0)
    # Apply Gaussian smoothing
    firing_rate_smoothed = gaussian_filter1d(firing_rate, sigma=sigma)
    # Create time bins (bin centers)
    num_bins = raster.shape[1]
    time_bins = np.linspace(0, sd.length, num_bins, endpoint=False) + bin_size/2
    return time_bins, firing_rate_smoothed


##############################################
# 5. Short-Time Synchrony Metric
##############################################

def synchrony_index(sd: SpikeData, window_size: float = 1000) -> (np.ndarray, np.ndarray):
    """
    Compute a synchrony index over time using a sliding window.
    For each window, the function computes the pairwise Pearson correlation (ignoring self-correlation)
    from the smoothed, binned spike raster.
    
    Parameters:
        sd (SpikeData): SpikeData object.
        window_size (float): Window size (ms) for computing synchrony.
    
    Returns:
        tuple: (time_points, synchrony_values) where:
            - time_points: center of each window (ms)
            - synchrony_values: average pairwise correlation within that window.
    """
    # Create a dense raster with a 1ms bin for high resolution.
    raster = sd.raster(bin_size=1)
    num_timepoints = raster.shape[1]
    step = int(window_size)  # move window in non-overlapping steps for simplicity
    synchrony_vals = []
    time_pts = []
    for start in range(0, num_timepoints - step + 1, step):
        window_raster = raster[:, start:start+step].astype(float)
        # Smooth the windowed raster if desired
        window_raster = gaussian_filter1d(window_raster, sigma=3, axis=1)
        corr_mat = np.corrcoef(window_raster)
        # Exclude self-correlations (diagonal) and compute the mean of upper triangle
        triu_indices = np.triu_indices_from(corr_mat, k=1)
        synchrony_vals.append(np.nanmean(corr_mat[triu_indices]))
        time_pts.append(start + step/2)
    return np.array(time_pts), np.array(synchrony_vals)


##############################################
# 6. Randomized Data Comparison Helper
##############################################

def randomized_metric(sd: SpikeData, metric_function, n_randomizations: int = 100, **kwargs):
    """
    Compute a given metric on the real spike data and compare it against the randomized data.
    The metric_function should take a SpikeData object as input and return a value or array.
    
    Parameters:
        sd (SpikeData): The real SpikeData object.
        metric_function (function): Function that computes a metric from a SpikeData object.
        n_randomizations (int): Number of randomizations to perform.
        kwargs: Additional keyword arguments to pass to the metric_function.
    
    Returns:
        tuple: (real_metric, mean_randomized, randomized_std)
    """
    real_metric = metric_function(sd, **kwargs)
    randomized_metrics = []
    for _ in range(n_randomizations):
        # Use the built-in randomize_raster method from SpikeData.
        randomized_train = sd.randomized(bin_size_ms=1)  # or use your own wrapper if needed
        # Create a new SpikeData object with randomized spike trains.
        randomized_sd = SpikeData(randomized_train, length=sd.length, N=sd.N,
                                  neuron_attributes=sd.neuron_attributes, metadata=sd.metadata)
        randomized_metrics.append(metric_function(randomized_sd, **kwargs))
    randomized_metrics = np.array(randomized_metrics)
    mean_rand = np.nanmean(randomized_metrics, axis=0)
    std_rand = np.nanstd(randomized_metrics, axis=0)
    return real_metric, mean_rand, std_rand


##############################################
# 7. Utility: Compare Lists Statistics
##############################################

def compare_statistical_lists(list1: list, list2: list) -> dict:
    """
    Compare two lists of numerical values by computing their mean, median, std, and performing a t-test.
    
    Parameters:
        list1 (list): First list of values.
        list2 (list): Second list of values.
    
    Returns:
        dict: Dictionary with summary statistics and t-test results.
    """
    from scipy.stats import ttest_ind
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    stats = {
        'mean1': np.nanmean(arr1),
        'mean2': np.nanmean(arr2),
        'median1': np.nanmedian(arr1),
        'median2': np.nanmedian(arr2),
        'std1': np.nanstd(arr1),
        'std2': np.nanstd(arr2),
    }
    try:
        t_stat, p_value = ttest_ind(arr1, arr2, nan_policy='omit')
        stats['t_stat'] = t_stat
        stats['p_value'] = p_value
    except Exception as e:
        stats['t_stat'] = np.nan
        stats['p_value'] = np.nan
    return stats


##############################################
# End of Module
##############################################

if __name__ == '__main__':
    # Example usage (assuming you have a valid SpikeData object 'sd'):
    # from load_npz import load_spikedata
    # sd = load_spikedata('path/to/your/auto_curated.zip')
    # rates = firing_rates(sd)
    # print("Mean firing rates:", rates)
    pass
