#!/usr/bin/env python3
"""
Module: compare_neurons_sd.py

This module provides functions for comparing spike trains between pairs of neurons.
The primary goal is to check, for two neurons, whether their spikes occur within a 
specified time window (in milliseconds) of each other.

Functions:
    1. count_coincident_spikes(spikes1, spikes2, window)
         - For each spike in spikes1, counts it as a coincidence if there is at least one 
           spike in spikes2 within ± window milliseconds.
    2. fraction_coincident_spikes(spikes1, spikes2, window)
         - Computes the fraction of spikes that are coincident between two spike trains,
           by averaging the fraction for spikes1 vs spikes2 and vice-versa.
    3. compare_neurons(sd, neuron_idx1, neuron_idx2, window)
         - Given a SpikeData object and two neuron indices, extracts their spike trains 
           and computes the fraction of coincident spikes using the above functions.
"""

import numpy as np

def count_coincident_spikes(spikes1, spikes2, window):
    """
    Count the number of spikes in spikes1 that have at least one spike in spikes2
    occurring within a specified time window.

    Parameters:
        spikes1 : np.ndarray
            1D array of spike times (in ms) for neuron 1.
        spikes2 : np.ndarray
            1D array of spike times (in ms) for neuron 2.
        window : float
            Time window (in ms). A spike in spikes1 is considered coincident if there is any 
            spike in spikes2 with an absolute time difference less than or equal to window.

    Returns:
        count : int
            The number of spikes in spikes1 with at least one coincident spike in spikes2.
    """
    count = 0
    for spike in spikes1:
        # Check if any spike in spikes2 is within window of the current spike.
        if np.any(np.abs(spikes2 - spike) <= window):
            count += 1
    return count

def fraction_coincident_spikes(spikes1, spikes2, window):
    """
    Compute the fraction of spikes that are coincident between two spike trains.
    
    This function calculates the fraction of spikes in spikes1 that are close (within window)
    to any spike in spikes2, and vice versa, and then returns the average of the two fractions.
    
    Parameters:
        spikes1 : np.ndarray
            Spike times (in ms) for neuron 1.
        spikes2 : np.ndarray
            Spike times (in ms) for neuron 2.
        window : float
            Time window (in ms) for considering spikes to be coincident.
    
    Returns:
        fraction : float
            The average fraction of coincident spikes between the two spike trains.
            Returns 0 if one of the trains is empty.
    """
    if len(spikes1) == 0 or len(spikes2) == 0:
        return 0.0
    
    count1 = count_coincident_spikes(spikes1, spikes2, window)
    fraction1 = count1 / len(spikes1)
    
    count2 = count_coincident_spikes(spikes2, spikes1, window)
    fraction2 = count2 / len(spikes2)
    
    return (fraction1 + fraction2) / 2

def compare_neurons(sd, neuron_idx1, neuron_idx2, window):
    """
    Compare the spike trains of two neurons in a SpikeData object to determine the 
    fraction of coincident spikes within a given time window.

    Parameters:
        sd : SpikeData
            A SpikeData object (spike times are assumed to be in milliseconds).
        neuron_idx1 : int
            Index of the first neuron.
        neuron_idx2 : int
            Index of the second neuron.
        window : float
            Time window (in ms) within which spikes are considered coincident.

    Returns:
        fraction : float
            The fraction of coincident spikes (average over both directions).
    """
    spikes1 = sd.train[neuron_idx1]
    spikes2 = sd.train[neuron_idx2]
    return fraction_coincident_spikes(spikes1, spikes2, window)

# For testing purposes:
if __name__ == '__main__':
    # Example test: two synthetic spike trains
    spikes1 = np.array([10, 50, 100, 150, 210])
    spikes2 = np.array([12, 55, 95, 152, 215])
    window = 5  # ms
    count = count_coincident_spikes(spikes1, spikes2, window)
    frac = fraction_coincident_spikes(spikes1, spikes2, window)
    print("Count of coincidences (spikes1 relative to spikes2):", count)
    print("Fraction of coincident spikes:", frac)
    
    # If you had a SpikeData object 'sd', you would do:
    # fraction = compare_neurons(sd, neuron_idx1=0, neuron_idx2=1, window=5)
    # print("Comparison fraction:", fraction)
