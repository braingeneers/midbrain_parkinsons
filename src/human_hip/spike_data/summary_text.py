#!/usr/bin/env python3

#### Summary ####
# This code contains creates a text summary of spike data (SD) files

import os
from human_hip.spike_data import read_phy_files


def summary_text(sd):
    """
    Function: provide a basic text summary of a spike data object
    Input: sd (Spike Data Object)- the standard type of  object used by braingeneers to store spike data
    Output: printed text summary giving, number of spikes/neurons, recording length, firing rate, and coefficient of variation
    """
    idces_control, times_control = sd.idces_times()
    n_neurons_control = len(sd.rates())

    print("Number of spikes: ", len(idces_control))
    print("Length: ", int(times_control[-1]/1000), "seconds")
    print("Number of Neurons: ", n_neurons_control)
    entire_firing_rate_control = len(idces_control) / (times_control[-1] / 1000)
    avg_rate_control = entire_firing_rate_control / n_neurons_control
    print("Average Firing Rate: ", round(avg_rate_control, 2))

    isis_raw = sd.interspike_intervals()
    # Remove all isi's greater than 100ms. As there are likely neurons not following periodic firing pattern
    isis = []
    for i in range(len(isis_raw)):
        isi = isis_raw[i]
        isis = isis + isi[isi < 100].tolist()

    isi_mean = sum(isis) / len(isis)
    isi_var = sum([((x - isi_mean) ** 2) for x in isis]) / len(isis)
    isi_std = isi_var ** 0.5
    cv = isi_std / isi_mean
    print("Coefficient of Variation: ", round(cv,3) )



def summary_UUID(folder_name, data_path="/workspaces/human_hippocampus/data/ephys" ):
    """
    Function: Provides a text summary of all spike data files in a UUID folder
    Inputs:
        folder_name (str): UUID folder name
        data_path (str): path to where the folder_name is located
    Output: 
        printed text summary giving, number of spikes/neurons, recording length, firing rate, and coefficient of variation
    """
    path = f"{data_path}/{folder_name}/derived/kilosort2/"
    spike_data_objects = {}  # Dictionary to store spike data objects

    for filename in os.listdir(path):
        if filename.endswith(".zip") and filename.__contains__("curated"):
            file_path = os.path.join(path, filename)
            try:
                sd = read_phy_files(file_path)
                sd.original_file = filename
                spike_data_objects[filename] = sd
            except:
                print(f"WARNING: Unable to Read < {filename} >")
    print("-----------------------------")       
    for sd_name, sd_object in spike_data_objects.items():
        print(f"Filename: {sd_name}:")
        summary_text(sd_object)
        print("-----------------------------")






