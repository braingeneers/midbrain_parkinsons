#!/usr/bin/env python3
"""
Module: sttc_sd.py

This module provides core functions to compute the spike time tiling coefficient (STTC)
across a list of spike trains. It includes:

    1. spike_time_tiling(tA, tB, delt=20, length=None)
         - Computes the STTC between two spike trains.
    2. compute_sttc_matrix(spike_trains, length, delt=20)
         - Computes the full pairwise STTC matrix for a list of spike trains given the recording length.
    3. get_upper_triangle(matrix, k=1)
         - Extracts and returns the values (and indices) in the upper triangle of a square matrix.
    4. compute_filtered_sttc(spike_trains, length, delt=20, n_shuffles=10, filter_method="difference")
         - Computes a filtered STTC matrix by randomizing the spike trains multiple times and then
           either subtracting (difference) or thresholding (only keeping positive differences) the
           randomized surrogate STTC matrix from the real STTC matrix.
           
Note: All spike times and the recording length are assumed to be in milliseconds.
"""

import numpy as np
import random as rand

# -----------------------------------------------------------------------------
def spike_time_tiling(tA, tB, delt=20, length=None):
    """
    Compute the spike time tiling coefficient (STTC) between two spike trains.
    
    Parameters:
        tA : np.ndarray
            Spike times for neuron A (in ms), assumed sorted.
        tB : np.ndarray
            Spike times for neuron B (in ms), assumed sorted.
        delt : float, optional
            Time window (ms) for the STTC calculation (default 20 ms).
        length : float, optional
            Duration of the recording (ms). If None, it is assumed to be the maximum of tA[-1] and tB[-1].
    
    Returns:
        sttc : float
            The computed spike time tiling coefficient.
    """
    if length is None:
        length = max(tA[-1] if tA.size > 0 else 0, tB[-1] if tB.size > 0 else 0)
    if tA.size == 0 or tB.size == 0:
        return 0.0

    # Helper: total time within delt of any spike in a train.
    def _sttc_ta(t, delt, tmax):
        if t.size == 0:
            return 0.0
        base = min(delt, t[0]) + min(delt, tmax - t[-1])
        return base + np.minimum(np.diff(t), 2 * delt).sum()

    TA = _sttc_ta(tA, delt, length) / length
    TB = _sttc_ta(tB, delt, length) / length

    # Helper: Count number of spikes within delt of any spike in the other train.
    def _count_in_delt(t1, t2, delt):
        if t2.size == 0:
            return 0
        idx = np.searchsorted(t2, t1)
        idx = np.clip(idx, 1, t2.size - 1)
        dt_left = np.abs(t2[idx] - t1)
        dt_right = np.abs(t2[idx - 1] - t1)
        return (np.minimum(dt_left, dt_right) <= delt).sum()

    PA = _count_in_delt(tA, tB, delt) / tA.size
    PB = _count_in_delt(tB, tA, delt) / tB.size

    val1 = (PA - TB) / (1 - PA * TB) if (1 - PA * TB) != 0 else 0
    val2 = (PB - TA) / (1 - PB * TA) if (1 - PB * TA) != 0 else 0
    sttc = (val1 + val2) / 2
    return sttc

# -----------------------------------------------------------------------------
def compute_sttc_matrix(spike_trains, length=None, delt=20):
    """
    Compute the full pairwise STTC matrix for a list of spike trains.
    
    Parameters:
        spike_trains : list of np.ndarray
            Each element is an array of spike times (in ms) for one neuron.
        length : float
            Duration of the recording (ms).
        delt : float, optional
            STTC window parameter (ms, default 20).
    
    Returns:
        matrix : np.ndarray
            A symmetric matrix of shape (N, N) where N is the number of neurons.
            The diagonal elements are set to 1.
    """
    N = len(spike_trains)
    matrix = np.zeros((N, N))
    for i in range(N):
        matrix[i, i] = 1.0
        for j in range(i + 1, N):
            sttc_val = spike_time_tiling(spike_trains[i], spike_trains[j], delt, length)
            matrix[i, j] = sttc_val
            matrix[j, i] = sttc_val
    return matrix

# -----------------------------------------------------------------------------
def get_upper_triangle(matrix, k=1):
    """
    Extract the upper triangle (above the k-th diagonal) of a square matrix.
    
    Parameters:
        matrix : np.ndarray
            A square matrix.
        k : int, optional
            Diagonal offset (k=1 excludes the main diagonal; default 1).
    
    Returns:
        values : np.ndarray
            A 1D array containing the values in the upper triangle.
        indices : tuple of np.ndarray
            The row and column indices corresponding to the extracted values.
    """
    indices = np.triu_indices_from(matrix, k=k)
    values = matrix[indices]
    return values, indices

# -----------------------------------------------------------------------------
def randomize_spike_trains(spike_trains, seed=None):
    """
    Randomize spike times for each neuron while preserving the total spike count.
    
    This function converts spike trains to a binary raster (using 1 ms bins), randomizes each
    neuron's spike locations, and converts back to spike time format.
    
    Parameters:
        spike_trains : list of np.ndarray
            Each element is an array of spike times (in ms) for one neuron.
        seed : int, optional
            Random seed for reproducibility.
    
    Returns:
        randomized_trains : list of np.ndarray
            Randomized spike trains.
    """
    rng = np.random.default_rng(seed)
    N = len(spike_trains)
    # Determine overall recording length from non-empty trains.
    length = max([np.max(ts) for ts in spike_trains if ts.size > 0])
    bin_size = 1  # ms
    num_bins = int(np.ceil(length / bin_size))
    
    # Build raster: rows are neurons, columns are time bins.
    raster = np.zeros((N, num_bins), dtype=int)
    for i, spikes in enumerate(spike_trains):
        if spikes.size > 0:
            indices = (spikes / bin_size).astype(int)
            raster[i, indices] = 1
    
    # Randomize each neuron's spike locations.
    randomized_raster = np.zeros_like(raster)
    for i in range(N):
        count = raster[i].sum()
        if count > 0:
            new_indices = rng.choice(num_bins, size=count, replace=False)
            randomized_raster[i, new_indices] = 1
    
    # Convert back to spike time format.
    randomized_trains = [np.where(randomized_raster[i])[0] * bin_size for i in range(N)]
    return randomized_trains
# -----------------------------------------------------------------------------
def best_effort_sample(counts, M, seed=None):
    """
    Sample M indices from 0...N-1 based on the provided counts without exceeding them.
    This is a simplified version.
    
    Parameters:
        counts : np.ndarray
            1D array of nonnegative integers.
        M : int
            Number of samples to draw.
        seed : int, optional
            Random seed.
    
    Returns:
        samples : list
            List of sampled indices.
    """
    rng = np.random.default_rng(seed)
    counts = counts.copy()
    samples = []
    available = np.arange(len(counts))
    while M > 0:
        # Compute probabilities proportional to counts.
        probs = counts[available] / counts[available].sum()
        chosen = rng.choice(available, p=probs, replace=False, size=1)[0]
        samples.append(chosen)
        counts[chosen] -= 1
        if counts[chosen] == 0:
            available = available[available != chosen]
        M -= 1
    return samples

def randomize_spike_trains_global(spike_trains, seed=5):
    """
    Randomize spike trains globally using a method similar to spikedata.py's randomize_raster.
    This function converts the spike trains into a binary raster (1 ms bins),
    then reallocates spikes across neurons (preserving overall spike counts per bin)
    using a best-effort sampling method, and finally converts back to spike time format.
    
    Parameters:
        spike_trains : list of np.ndarray
            Each element is an array of spike times (in ms) for one neuron.
        seed : int, optional
            Random seed.
    
    Returns:
        randomized_trains : list of np.ndarray
            The globally randomized spike trains.
    """
    rng = np.random.default_rng(seed)
    N = len(spike_trains)
    # Determine overall recording length from non-empty trains.
    length = max([np.max(ts) for ts in spike_trains if ts.size > 0])
    bin_size = 1  # ms
    num_bins = int(np.ceil(length / bin_size))
    
    # Create binary raster: rows=neurons, columns=time bins.
    raster = np.zeros((N, num_bins), dtype=int)
    for i, spikes in enumerate(spike_trains):
        if spikes.size > 0:
            indices = (spikes / bin_size).astype(int)
            raster[i, indices] = 1

    # Initialize an empty raster for the randomized result.
    randomized_raster = np.zeros_like(raster)
    # For each time bin, reassign the spikes randomly across neurons.
    # Get the number of spikes in each bin.
    spikes_per_bin = raster.sum(axis=0)
    bin_order = np.argsort(spikes_per_bin)[::-1]  # Process bins in descending order of spike count.
    bin_order = bin_order[spikes_per_bin[bin_order] > 0]
    
    # Sum of spikes per neuron (weights) from the entire raster.
    neuron_weights = raster.sum(axis=1)
    for b in bin_order:
        M = spikes_per_bin[b]
        # Use best_effort_sample to choose neurons for these spikes.
        chosen_units = best_effort_sample(neuron_weights, M, rng)
        for unit in chosen_units:
            neuron_weights[unit] = max(neuron_weights[unit]-1, 0)
            randomized_raster[unit, b] += 1

    # Convert randomized raster back to spike times.
    randomized_trains = [np.where(randomized_raster[i])[0] * bin_size for i in range(N)]
    return randomized_trains
#-----------------------------------------------------------------------------
def random_rotation(spike_trains, seed=1):
    """
    Randomizes a list of spike trains by rotating each neuron's spike train.
    
    For each spike train:
      - A random cut point is chosen (between index 1 and len(train)-1).
      - The train is rotated by taking the segment from the cut point to the end,
        shifting it so that its first spike becomes 0, and then appending the earlier
        segment shifted by an offset so that the overall timing is preserved.
    
    Parameters:
        spike_trains : list of np.ndarray
            A list where each element is a 1D NumPy array of spike times (in ms) for one neuron.
            Spike times are assumed to be sorted in ascending order.
        seed : int, optional
            Seed for reproducibility (default is 1).
    
    Returns:
        rotated_trains : list of np.ndarray
            A list of rotated spike trains.
    """
    import numpy as np
    import random as rand

    rotated_trains = []
    # Set a global seed and create a per-train seed array for reproducibility.
    np.random.seed(seed)
    seeds = [np.random.randint(0, 1000000000) for _ in range(len(spike_trains))]
    
    for i, train in enumerate(spike_trains):
        # If the train has fewer than 2 spikes, rotation is not applicable.
        if train.size < 2:
            rotated_trains.append(train)
            continue
        
        # Set local seed for this train.
        rand.seed(seeds[i])
        # Choose a random cut point between 1 and len(train)-1.
        alpha = rand.randrange(1, len(train))
        
        # Rotate the spike train:
        # Segment 1: from index 'alpha' to end, shifted so that its first spike is 0.
        first_segment = train[alpha:] - train[alpha]
        # Compute offset so that the second segment continues after the first.
        offset = first_segment[-1] + (train[alpha] - train[0])
        # Segment 2: from index 0 to alpha, shifted by the computed offset.
        second_segment = train[:alpha] + offset
        
        # Concatenate the segments to form the rotated train.
        rotated = np.concatenate((first_segment, second_segment))
        rotated_trains.append(rotated)
    
    return rotated_trains

def filter(true_matrix, threshold, percentile= None, sttc_array=None):
    """
    Compute the 95th percentile (across iterations) for each neuron pair,
    then retain the true STTC value only if it exceeds that percentile.
    """
    if sttc_array is None:
        filtered = np.where(true_matrix >= threshold, true_matrix, 0)
        return filtered
    if sttc_array is not None:
        percentiled = np.percentile(sttc_array, percentile, axis=0)
        filtered = np.where(true_matrix >= percentiled, true_matrix, 0)
        return filtered


# -----------------------------------------------------------------------------
def compute_filtered_sttc(spike_trains, delt=20, n_shuffles=3, length=None,
                          filter_method=None, randomization_method=None):
    """
    Compute a filtered STTC matrix using randomized surrogate spike trains.
    
    Parameters:
        spike_trains : list of np.ndarray
            Spike trains (in ms) for each neuron.
        length : float
            Recording duration (ms).
        delt : float, optional
            STTC window parameter (default 20 ms).
        n_shuffles : int, optional
            Number of randomizations to perform (default 10).
        filter_method : str, optional
            "difference": returns (real STTC - surrogate average), which may be negative.
            "threshold": returns the difference with negatives set to zero.
        randomization_method : str, optional
            "local" to use the neuron-by-neuron randomization (default),
            "global" to use the spikedata-style randomization,
            "rotation" to use the random rotation method.
    
    Returns:
        filtered_sttc : np.ndarray
            The filtered STTC matrix.
    """
    # Compute real STTC matrix.

    real_sttc = compute_sttc_matrix(spike_trains, delt, length)
    
    # Choose randomization method.
    if randomization_method == "local":
        rand_func = randomize_spike_trains  # existing local randomization function
    elif randomization_method == "global":
        rand_func = randomize_spike_trains_global  # the global version 
    elif randomization_method == "rotation":
        rand_func = random_rotation #rotation method
    else:
        raise ValueError("Invalid randomization_method. Choose 'local' or 'global'.")

    # Compute surrogate (randomized) STTC matrices.
    surrogate_matrices = []
    for _ in range(n_shuffles):
        randomized_trains = rand_func(spike_trains)
        surrogate = compute_sttc_matrix(randomized_trains, length, delt)
        surrogate_matrices.append(surrogate)
    surrogate_matrices = np.array(surrogate_matrices)
    surrogate_avg = np.mean(surrogate_matrices, axis=0)
    
    # Compute filtered STTC.
    diff = real_sttc - surrogate_avg
    if filter_method == "difference":
        filtered_sttc = diff
    elif filter_method == "threshold":
        filtered_sttc = np.where(diff > 0, diff, 0)

    else:
        raise ValueError("Invalid filter_method. Choose 'difference' or 'threshold'.")
    
    return filtered_sttc


# -----------------------------------------------------------------------------
# For testing purposes:
if __name__ == '__main__':
    # Create synthetic spike trains (in ms)
    np.random.seed(42)
    # For example, 3 neurons with 50 random spike times between 0 and 1000 ms.
    spike_trains = [np.sort(np.random.randint(0, 1000, size=50)) for _ in range(3)]
    length = 1000  # ms
    delt = 20    # ms
    
    sttc_mat = compute_sttc_matrix(spike_trains, length, delt)
    print("Raw STTC Matrix:")
    print(sttc_mat)
    
    upper_vals, idx = get_upper_triangle(sttc_mat)
    print("Upper triangle values:")
    print(upper_vals)
    
    filtered_diff = compute_filtered_sttc(spike_trains, length, delt, n_shuffles=10, filter_method="difference")
    filtered_thresh = compute_filtered_sttc(spike_trains, length, delt, n_shuffles=10, filter_method="threshold")
    print("Filtered STTC Matrix (difference method):")
    print(filtered_diff)
    print("Filtered STTC Matrix (threshold method):")
    print(filtered_thresh)
