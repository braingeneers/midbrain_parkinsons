#!/usr/bin/env python3
"""
Module: load_npz.py
Purpose: Load auto-curated NPZ data (inside a ZIP) and return a SpikeData object.
"""

import zipfile
import numpy as np
import smart_open

from braingeneers.analysis.analysis import SpikeData


def reformat_neuron_data(neuron_data):
    """
    Reformat neuron_data from a dict mapping neuron IDs to metadata dictionaries
    into a dict mapping each metadata field to a list of values (one per neuron),
    sorted by neuron ID.
    
    Example:
        Input: { 0: {'position': [x0, y0], 'amplitude': amp0},
                 1: {'position': [x1, y1], 'amplitude': amp1}, ... }
        Output: {'position': [[x0, y0], [x1, y1], ...],
                 'amplitude': [amp0, amp1, ...]}
    
    Parameters:
        neuron_data : dict
            Original metadata dictionary from the NPZ file.
            
    Returns:
        new_data : dict
            Reformatted metadata with each field as a list of length N.
    """
    sorted_keys = sorted(neuron_data.keys())
    # Get the field names from the first neuron's metadata
    first_meta = neuron_data[sorted_keys[0]]
    fields = list(first_meta.keys())
    new_data = {field: [] for field in fields}
    for k in sorted_keys:
        meta = neuron_data[k]
        for field in fields:
            new_data[field].append(meta.get(field))
    return new_data


def load_spikedata(qm_path, read_config=False):
    """
    Load an auto-curated NPZ file from a ZIP archive and convert it into a SpikeData object.
    
    Parameters:
        qm_path (str): Path to the ZIP archive containing "qm.npz".
        read_config (bool): Whether to read and store the configuration data from the NPZ.
    
    Returns:
        SpikeData: A SpikeData object with spike times in milliseconds.
    
    Note:
        The NPZ file must contain:
          - "train": a dict of spike times (auto-curated)
          - "fs": sampling rate (Hz)
          - "neuron_data": neuron metadata/attributes (a dict mapping neuron IDs to metadata)
          - Optionally, "config": additional configuration parameters.
    """
    with smart_open.open(qm_path, 'rb') as f:
        with zipfile.ZipFile(f, 'r') as f_zip:
            with f_zip.open("qm.npz") as qm_file:
                data = np.load(qm_file, allow_pickle=True)
                spike_times = data["train"].item()
                fs = data["fs"]
                # Convert each neuron's spike times (divide by sampling rate)
                train = [times / fs for _, times in spike_times.items()]
                config = data["config"].item() if read_config else None
                neuron_data = data["neuron_data"].item()

    # Convert spike times from seconds to milliseconds
    train_ms = [a_train * 1000 for a_train in train]
    
    # Determine the recording duration (in ms)
    sd_auto_length = max(max(a_train) for a_train in train_ms if len(a_train) > 0)
    
    # Prepare metadata; include config if available
    metadata = {}
    if config is not None:
        metadata['config'] = config

    # Reformat the neuron metadata to a dictionary of lists.
    new_neuron_data = reformat_neuron_data(neuron_data)
    
    return SpikeData(
        train_ms, 
        length=sd_auto_length, 
        N=len(train_ms),
        neuron_data=new_neuron_data, 
        neuron_attributes=new_neuron_data,
        metadata=metadata
    )


# For testing purposes:
if __name__ == '__main__':
    # Example usage:
    # sd = load_spikedata("path/to/your/data.zip", read_config=True)
    # print(sd.neuron_data)  # Should now be a dict of lists with each field having length equal to number of neurons.
    pass
