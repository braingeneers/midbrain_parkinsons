#!/usr/bin/env python3
"""
Module: utils_sd.py

This module contains general utility functions for data analysis.
One function here scans neuron positions and identifies neurons that share the same position.
"""

import numpy as np
from positions_sd import get_neuron_positions  # assuming this function exists

def find_duplicate_positions(sd, decimals=6):
    """
    Identify neurons that have (nearly) the same position.

    Parameters:
        sd : SpikeData
            A SpikeData object that contains neuron position information.
        decimals : int, optional
            The number of decimal places to round positions to when comparing (default 6).

    Returns:
        duplicates : dict
            Dictionary where keys are position tuples (x, y) and values are lists of neuron indices 
            that have that same position. Only positions with more than one neuron are included.
    """
    positions = get_neuron_positions(sd)  # Returns an (N,2) array of positions
    duplicates = {}
    
    # Iterate over all positions
    for idx, pos in enumerate(positions):
        # Round the position to the given number of decimals and convert to a tuple.
        pos_key = tuple(np.round(pos, decimals=decimals))
        if pos_key in duplicates:
            duplicates[pos_key].append(idx)
        else:
            duplicates[pos_key] = [idx]
    
    # Filter out keys with only a single neuron (i.e., no duplicates)
    duplicates = {pos: indices for pos, indices in duplicates.items() if len(indices) > 1}
    return duplicates

# For testing purposes:
if __name__ == '__main__':
    # Example usage (assuming you have a valid SpikeData object 'sd'):
    # from load_npz import load_spikedata
    # sd = load_spikedata('path/to/your/auto_curated.zip')
    # dupes = find_duplicate_positions(sd)
    # print("Duplicate neuron positions:")
    # for pos, indices in dupes.items():
    #     print(f"Position {pos}: Neuron indices {indices}")
    pass
