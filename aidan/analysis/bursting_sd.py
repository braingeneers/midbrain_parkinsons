#!/usr/bin/env python3
"""
Module: bursting_sd.py

This module provides functions for analyzing bursting dynamics from a SpikeData object.
It leverages existing methods from spikedata.py (such as:
    - burst_detection
    - burstiness_index
    - avalanches / avalanche_duration_size
    - deviation_from_criticality
) and adds additional functions for:
    • Fitting power law distributions to avalanche sizes.
    • Building a network (using STTC as a proxy for burst-based connectivity) and computing basic network metrics.
    
No plotting is performed here—only numerical results and data structures are returned.
"""

import numpy as np
from braingeneers.analysis.analysis import (
    SpikeData,
    burst_detection,
    burstiness_index,
    avalanche_duration_size,
    deviation_from_criticality
)

# For power law fitting, we use the powerlaw library.
try:
    import powerlaw
except ImportError:
    powerlaw = None

# For network analysis, we use networkx.
import networkx as nx


#############################
# 1. Basic Bursting Metrics
#############################

def burst_detection(spike_times, burst_threshold, spike_num_thr=3):
    """
    Detect burst from spike times with a interspike interval threshold (burst_threshold)
    and a spike number threshold (spike_num_thr).

    Returns:
        spike_num_list -- a list of burst features
          [index of burst start point, number of spikes in this burst]
        burst_set -- a list of spike times of all the bursts.
    """
    # TODO missing tests
    spike_num_burst = 1
    spike_num_list = []
    for i in range(len(spike_times) - 1):
        if spike_times[i + 1] - spike_times[i] <= burst_threshold:
            spike_num_burst += 1
        else:
            if spike_num_burst >= spike_num_thr:
                spike_num_list.append([i - spike_num_burst + 1, spike_num_burst])
                spike_num_burst = 1
            else:
                spike_num_burst = 1
    burst_set = []
    for loc in spike_num_list:
        for i in range(loc[1]):
            burst_set.append(spike_times[loc[0] + i])
    return spike_num_list, burst_set


def compute_burst_index(sd, bin_size=40):
    """
    Compute the burstiness index from the SpikeData object.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        bin_size : float
            Bin size (in ms) used for the burstiness computation.
    
    Returns:
        burst_index : float
            A value between 0 and 1 indicating the overall burstiness.
    """
    return sd.burstiness_index(bin_size)


def get_avalanches(sd, quantile=0.35, bin_size=40):
    """
    Detect avalanches from the SpikeData object.
    
    This function uses the binned spike counts to determine a threshold
    (using the specified quantile) and then extracts avalanche durations and sizes.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        quantile : float
            Quantile (0 to 1) used to compute the threshold for avalanche detection.
        bin_size : float
            Bin size (in ms) for binning spike times.
    
    Returns:
        durations : np.ndarray
            Array of avalanche durations (number of bins).
        sizes : np.ndarray
            Array of avalanche sizes (total spike count within each avalanche).
    """
    # The binned spike counts are obtained via the built-in method:
    binned_counts = sd.binned(bin_size)
    threshold = np.quantile(binned_counts, quantile)
    durations, sizes = avalanche_duration_size(sd, threshold, bin_size)
    return durations, sizes


def summarize_burst_detection(sd, burst_threshold, spike_num_thr=3):
    """
    Run burst detection on each neuron's spike train.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        burst_threshold : float
            The maximum interspike interval (in ms) that defines a burst.
        spike_num_thr : int, optional
            Minimum number of spikes required to qualify as a burst.
    
    Returns:
        burst_summary : dict
            A dictionary mapping neuron indices to a tuple
            (burst_features, burst_spike_times) as returned by burst_detection.
    """
    burst_summary = {}
    for i, neuron_spikes in enumerate(sd.train):
        if len(neuron_spikes) == 0:
            burst_summary[i] = ([], [])
        else:
            bursts, burst_spikes = burst_detection(neuron_spikes, burst_threshold, spike_num_thr)
            burst_summary[i] = (bursts, burst_spikes)
    return burst_summary


def compute_dcc(sd, quantile=0.35, bin_size=40, N_surrogate=1000, pval_truncated=0.05):
    """
    Compute the Deviation from Criticality (DCC) metric from the SpikeData object.
    
    This function wraps the built-in deviation_from_criticality method.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        quantile : float
            Quantile for computing the avalanche threshold.
        bin_size : float
            Bin size (in ms) used in the analysis.
        N_surrogate : int
            Number of surrogate datasets for the statistical test.
        pval_truncated : float
            p-value cutoff when testing a truncated power-law fit.
    
    Returns:
        dcc_result : namedtuple
            A named tuple (dcc, p_size, p_duration) containing the DCC metric and associated p-values.
    """
    return sd.deviation_from_criticality(quantile=quantile, bin_size=bin_size,
                                          N=N_surrogate, pval_truncated=pval_truncated)


#############################
# 2. Power Law Fitting for Avalanches
#############################

def fit_avalanche_power_law(sd, quantile=0.35, bin_size=40, **kwargs):
    """
    Fit a power law to the avalanche sizes extracted from the SpikeData object.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        quantile : float
            Quantile used to set the threshold for avalanche detection.
        bin_size : float
            Bin size (in ms) for binning spike times.
        kwargs : dict
            Additional keyword arguments to pass to powerlaw.Fit (if any).
    
    Returns:
        fit_results : dict
            A dictionary containing the fitted exponent ('alpha'), xmin ('xmin'),
            and optionally other parameters from the powerlaw fit.
            
    Note:
        Requires the powerlaw package.
    """
    if powerlaw is None:
        raise ImportError("The powerlaw package is required for power law fitting.")
    
    # Use the avalanche detection function to get avalanche sizes.
    _, sizes = get_avalanches(sd, quantile=quantile, bin_size=bin_size)
    avalanche_sizes = np.array(sizes)
    
    # Remove zeros or negative values (if any) before fitting.
    avalanche_sizes = avalanche_sizes[avalanche_sizes > 0]
    
    if avalanche_sizes.size == 0:
        raise ValueError("No avalanche sizes available for power law fitting.")
    
    # Fit the power law
    fit = powerlaw.Fit(avalanche_sizes, **kwargs)
    fit_results = {
        'alpha': fit.power_law.alpha,
        'xmin': fit.power_law.xmin,
        'D': fit.power_law.D  # KS distance
    }
    # Optionally, compare with a truncated power law if desired:
    if hasattr(fit, 'distribution_compare'):
        R, p = fit.distribution_compare('power_law', 'truncated_power_law', nested=True)
        fit_results['R'] = R
        fit_results['p'] = p
    return fit_results


#############################
# 3. Network Analysis Based on Burst-Related Connectivity
#############################

def build_burst_network(sd, threshold=0.5):
    """
    Build a network graph from the SpikeData object based on STTC values,
    which can serve as a proxy for burst-based connectivity.
    
    Only edges with STTC values above the threshold are included.
    
    Parameters:
        sd : SpikeData
            A SpikeData object.
        threshold : float
            Minimum STTC value required to include an edge.
    
    Returns:
        G : networkx.Graph
            A graph where nodes are neurons and an edge exists if the STTC between neurons exceeds the threshold.
        metrics : dict
            A dictionary of basic network metrics (average degree, clustering coefficient, etc.).
    """
    # Compute the full STTC matrix using the SpikeData method.
    sttc_mat = sd.spike_time_tiling()  # Assumes this returns an (N x N) matrix.
    
    N = sttc_mat.shape[0]
    G = nx.Graph()
    # Add all nodes.
    for i in range(N):
        G.add_node(i)
    
    # Add edges for pairs with STTC above threshold.
    for i in range(N):
        for j in range(i+1, N):
            if sttc_mat[i, j] >= threshold:
                G.add_edge(i, j, weight=sttc_mat[i, j])
    
    # Compute some basic network metrics.
    avg_degree = sum(dict(G.degree()).values()) / float(len(G.nodes()))
    clustering = nx.average_clustering(G, weight='weight') if len(G.nodes()) > 0 else np.nan
    metrics = {
        'average_degree': avg_degree,
        'average_clustering': clustering,
        'number_of_nodes': G.number_of_nodes(),
        'number_of_edges': G.number_of_edges()
    }
    return G, metrics


#############################
# End of Module
#############################

if __name__ == '__main__':
    # Example usage (assuming you have a valid SpikeData object 'sd'):
    # from load_npz import load_spikedata
    # sd = load_spikedata('path/to/your/auto_curated.zip')
    #
    # burst_idx = compute_burst_index(sd, bin_size=40)
    # print("Burstiness index:", burst_idx)
    #
    # durations, sizes = get_avalanches(sd, quantile=0.35, bin_size=40)
    # print("Avalanche durations:", durations)
    # print("Avalanche sizes:", sizes)
    #
    # burst_summary = summarize_burst_detection(sd, burst_threshold=50, spike_num_thr=3)
    # print("Burst summary for neuron 0:", burst_summary.get(0))
    #
    # backbone = compute_backbone_units(sd, criterion='x_threshold', threshold=500)
    # print("Backbone units mask:", backbone)
    #
    # dcc_result = compute_dcc(sd, quantile=0.35, bin_size=40)
    # print("DCC result:", dcc_result)
    #
    # if powerlaw is not None:
    #     pl_results = fit_avalanche_power_law(sd, quantile=0.35, bin_size=40)
    #     print("Power law fit results:", pl_results)
    #
    # G, net_metrics = build_burst_network(sd, threshold=0.5)
    # print("Network metrics:", net_metrics)
    pass
