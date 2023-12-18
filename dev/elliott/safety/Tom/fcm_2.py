#!/usr/bin/env python3



#from human_hip import basics
#def test_function2( to_print ):
#    basics.test_function( to_print )

# Importing libraries
import os # For file path
import glob # For file path
import random # For random sampling
import numpy as np # For data manipulation
import scipy # For data manipulation
import scipy.io as sio # For data manipulation
import scipy.ndimage as ndimage # For data manipulation
import pandas as pd # For data manipulation
import matplotlib.pyplot as plt # For plotting
import matplotlib.colors as mcolors # For plotting
import matplotlib.patches as patches # For plotting
from matplotlib.lines import Line2D # For plotting
from PIL import Image # For plotting
from ipywidgets import interact, interactive, fixed, interact_manual # For plotting
import braingeneers # For accessing braingeneers functions and classes
import braingeneers.data.datasets_electrophysiology as ephys # For accessing ephys functions and classes
from braingeneers.analysis.analysis import SpikeData, read_phy_files # For accessing SpikeData and read_phy_files

"""
Parameters
----------
dataset_path : str
    Path to the dataset
start : int
    Start time in seconds
stop : int
    Stop time in seconds
name : str
    Name of the dataset
latency_thresh : float
    Threshold for latency
latencies_ms_thresh : int
    Threshold for latencies in milliseconds
line_threshold : float
    Threshold for line
saved : str
    Whether to save the plot or not

Returns
-------
savedFCMplot : matplotlib.pyplot.scatter
    The plot of the FCM

Examples
--------
>>> FCM_Plotter("/home/jovyan/work/Human_Hippocampus/data/hippocampus/hippocampus_1", 0, 10, "hippocampus_1", 0.2, 100, 0.5, "yes")
"""

# Function to plot FCM
def FCM_Plotter(dataset_path, start, stop, name, latency_thresh=0.2, latencies_ms_thresh=100, line_threshold=0.5, saved='yes'):
    sd = read_phy_files(dataset_path)
    sd_start = sd.subtime(start*1000, stop*1000)

    not_empties = []
    empties = []
    arrays = sd_start.train
    
    # Find empty arrays
    for i, arr in enumerate(arrays):
        if len(arr) > 0:
            not_empties.append(i)
        if len(arr) == 0:
            empties.append(i)
    # Subset the arrays
    sub_start = sd_start.subset(not_empties)
    # Find the latencies
    def latencies_mean(lat_list):
        nested_list = lat_list
        for i in range(len(nested_list)):
            sublist = nested_list[i]
            length = len(sublist)
            if length == 0:
                sublist_mean = 0
            else:
                sublist_mean = sum(sublist) / len(sublist)
                sublist_mean = round(sublist_mean, 3)  # Round to 3d.p.
            nested_list[i] = sublist_mean
        return nested_list
    # Calculate the mean latencies
    def calculate_mean_latencies(sd, latencies_ms_thresh):
        num_neurons = sd.N
        latencies_array = [None] * num_neurons

        for curr_neuron in range(num_neurons):
            latencies = latencies_mean(sd.latencies_to_index(curr_neuron, window_ms=latencies_ms_thresh))
            latencies_array[curr_neuron] = latencies

        return latencies_array # Returns a list of lists
    # Calculate the mean latencies
    start_latencies = calculate_mean_latencies(sub_start, latencies_ms_thresh)
    # Calculate the in and out degree
    def compute_in_out_degree(latencies_array):
        num_neurons = len(latencies_array)
        in_out_deg = [(0, 0) for _ in range(num_neurons)]
        # in_out_deg = [None] * num_neurons
        for curr_neuron in range(num_neurons):
            in_deg = 0
            out_deg = 0
            curr_neural_latencies = latencies_array[curr_neuron]
            # in_out_deg[curr_neuron] = (in_deg, out_deg)
            for i in range(len(curr_neural_latencies)):
                if curr_neural_latencies[i] > 0:
                    out_deg += 1
                if curr_neural_latencies[i] < 0:
                    in_deg += 1

            in_out_deg[curr_neuron] = (in_deg, out_deg)

        return in_out_deg # Returns a list of tuples
    # Calculate the in and out degree
    start_in_out_deg = compute_in_out_degree(start_latencies)
    # Label the nodes
    def label_nodes(in_out_deg, latency_thresh=0.2):
        node_info = ['grey'] * len(in_out_deg)
        # node_info = [None] * len(in_out_deg)
        for i in range(len(in_out_deg)):
            test1 = (in_out_deg[i][1] - in_out_deg[i][0]) / (in_out_deg[i][1] + in_out_deg[i][0])
            test2 = (in_out_deg[i][0] - in_out_deg[i][1]) / (in_out_deg[i][1] + in_out_deg[i][0])
            # node_info[i] = (test1, test2)
            if test1 > latency_thresh:
                node_info[i] = 'red'
            if test2 > latency_thresh:
                node_info[i] = 'blue'

        return node_info # Returns a list of strings
    # Label the nodes
    colors = label_nodes(start_in_out_deg, latency_thresh)
    # Plot the FCM
    def closest_value(number):
        closest = 5
        if abs(number - 20) < abs(number - closest):
            closest = 20
        if abs(number - 50) < abs(number - closest):
            closest = 50
        return closest
    # Remove empty neurons
    sub_start.neuron_data = sd_start.neuron_data
    neur_data = sub_start.neuron_data[0]
    for key in empties:
        del neur_data[key]
    sub_start.neuron_data[0] = neur_data
    # Plot the FCM
    def sttc_neuron_plotter(inp_sd, upd_node_info, line_threshold):
        neuron_x = []
        neuron_y = []
        neuron_amp = []
        # Plot the FCM
        for neuron in inp_sd.neuron_data[0].values():
            neuron_x.append(neuron['position'][0])
            neuron_y.append(neuron['position'][1])
            neuron_amp.append(np.mean(neuron['amplitudes']))

        neuron_amp = [closest_value(num) for num in neuron_amp]
        # Plot the FCM
        plt.figure(figsize=(8, 6))
        savedFCMplot = plt.scatter(neuron_x, neuron_y, s=neuron_amp, c=upd_node_info)
        # Set fixed limits for x and y axes
        threshold = line_threshold
        sttc = inp_sd.spike_time_tilings()

        for i in range(sttc.shape[0]):
            for j in range(sttc.shape[1]):
                if i <= j:
                    continue
                if sttc[i, j] < threshold:
                    continue
                if i in empties:
                    continue
                if j in empties:
                    continue
                ix, iy = inp_sd.neuron_data[0][i]['position']
                jx, jy = inp_sd.neuron_data[0][j]['position']
                linewidth = 1.5 + 2 * (sttc[i, j] - threshold)
                opacity = 0.2 + 0.8 * (sttc[i, j] - threshold)
                plt.plot([ix, jx], [iy, jy], linewidth=linewidth, c='grey', alpha=opacity)

        plt.xlabel('um')
        plt.ylabel('um')
        plt.title(f"{name}")  # Adding the title

        # Set fixed limits for x and y axes
        plt.xlim(600, 2000)
        plt.ylim(0, 2200)
        # Set fixed limits for x and y axes
        node_degree_legend_elements = [
            plt.scatter([], [], s=5, marker='o', edgecolor='black', facecolor='none', label='5'),
            plt.scatter([], [], s=20, marker='o', edgecolor='black', facecolor='none', label='20'),
            plt.scatter([], [], s=50, marker='o', edgecolor='black', facecolor='none', label='50')
        ]
        # Set fixed limits for x and y axes
        node_type_legend_elements = [
            plt.scatter([], [], s=50, marker='o', edgecolor='black', facecolor='grey', label='Broker'),
            plt.scatter([], [], s=50, marker='o', edgecolor='black', facecolor='red', label='Sender'),
            plt.scatter([], [], s=50, marker='o', edgecolor='black', facecolor='blue', label='Receiver')
        ]
        # Set fixed limits for x and y axes
        node_degree_legend = plt.legend(handles=node_degree_legend_elements, title='Node Degree', loc='lower right')
        plt.gca().add_artist(node_degree_legend)

        correlation_legend_elements = [
            plt.Line2D([0], [0], color='grey', linewidth=0.5, label='0.6'),
            plt.Line2D([0], [0], color='grey', linewidth=1.0, label='0.8'),
            plt.Line2D([0], [0], color='grey', linewidth=1.5, label='1.0')
        ]

        correlation_legend = plt.legend(handles=correlation_legend_elements, title='Correlation', loc='lower left')
        plt.gca().add_artist(correlation_legend)

        node_type_legend = plt.legend(handles=node_type_legend_elements, title='Node Type', loc='best')
        if saved.lower() == "yes":
            # save plot to folder
            savepath = "/home/jovyan/work/Human_Hippocampus/saved_plots/fcm/" + f"{name}_{start}_to_{stop}sec.png"
            plt.savefig(savepath)
            plt.close()
        elif saved.lower() == "return":
            # saves plot to variable
            return savedFCMplot
        else:
            # plots locally 
            return
        
    sttc_neuron_plotter(sd, colors,line_threshold)
