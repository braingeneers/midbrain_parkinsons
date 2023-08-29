import os
import re
import glob
import random
import numpy as np
import scipy
import scipy.io as sio
import scipy.ndimage as ndimage
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image
from ipywidgets import interact, interactive, fixed, interact_manual
import braingeneers
import braingeneers.data.datasets_electrophysiology as ephys
from braingeneers.analysis.analysis import SpikeData, read_phy_files

def FCM_Plotter(dataset_path, start, stop, name, latency_thresh=0.2, latencies_ms_thresh=100, line_threshold=0.5, saved='yes'):
    sd = read_phy_files(dataset_path)
    sd_start = sd.subtime(start*1000, stop*1000)

    not_empties = []
    empties = []
    arrays = sd_start.train
    

    for i, arr in enumerate(arrays):
        if len(arr) > 0:
            not_empties.append(i)
        if len(arr) == 0:
            empties.append(i)

    sub_start = sd_start.subset(not_empties)

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

    def calculate_mean_latencies(sd, latencies_ms_thresh):
        num_neurons = sd.N
        latencies_array = [None] * num_neurons

        for curr_neuron in range(num_neurons):
            latencies = latencies_mean(sd.latencies_to_index(curr_neuron, window_ms=latencies_ms_thresh))
            latencies_array[curr_neuron] = latencies

        return latencies_array

    start_latencies = calculate_mean_latencies(sub_start, latencies_ms_thresh)

    def compute_in_out_degree(latencies_array):
        num_neurons = len(latencies_array)
        in_out_deg = [(0, 0) for _ in range(num_neurons)]

        for curr_neuron in range(num_neurons):
            in_deg = 0
            out_deg = 0
            curr_neural_latencies = latencies_array[curr_neuron]

            for i in range(len(curr_neural_latencies)):
                if curr_neural_latencies[i] > 0:
                    out_deg += 1
                if curr_neural_latencies[i] < 0:
                    in_deg += 1

            in_out_deg[curr_neuron] = (in_deg, out_deg)

        return in_out_deg

    start_in_out_deg = compute_in_out_degree(start_latencies)

    def label_nodes(in_out_deg, latency_thresh=0.2):
        node_info = ['grey'] * len(in_out_deg)

        for i in range(len(in_out_deg)):
            test1 = (in_out_deg[i][1] - in_out_deg[i][0]) / (in_out_deg[i][1] + in_out_deg[i][0])
            test2 = (in_out_deg[i][0] - in_out_deg[i][1]) / (in_out_deg[i][1] + in_out_deg[i][0])

            if test1 > latency_thresh:
                node_info[i] = 'red'
            if test2 > latency_thresh:
                node_info[i] = 'blue'

        return node_info

    colors = label_nodes(start_in_out_deg, latency_thresh)

    def closest_value(number):
        closest = 5
        if abs(number - 20) < abs(number - closest):
            closest = 20
        if abs(number - 50) < abs(number - closest):
            closest = 50
        return closest

    sub_start.neuron_data2 = sd_start.neuron_data
    neur_data = sub_start.neuron_data2[0]
    for key in empties:
        del neur_data[key]
    sub_start.neuron_data2[0] = neur_data

    def sttc_neuron_plotter(inp_sd, upd_node_info, line_threshold):
        neuron_x = []
        neuron_y = []
        neuron_amp = []

        for neuron in inp_sd.neuron_data2[0].values():
            neuron_x.append(neuron['position'][0])
            neuron_y.append(neuron['position'][1])
            neuron_amp.append(np.mean(neuron['amplitudes']))

        neuron_amp = [closest_value(num) for num in neuron_amp]

        plt.figure(figsize=(8, 6))
        savedFCMplot = plt.scatter(neuron_x, neuron_y, s=neuron_amp, c=upd_node_info)

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
                ix, iy = inp_sd.neuron_data2[0][i]['position']
                jx, jy = inp_sd.neuron_data2[0][j]['position']
                linewidth = 1.5 + 2 * (sttc[i, j] - threshold)
                opacity = 0.2 + 0.8 * (sttc[i, j] - threshold)
                plt.plot([ix, jx], [iy, jy], linewidth=linewidth, c='grey', alpha=opacity)

        plt.xlabel('um')
        plt.ylabel('um')
        plt.title(f"{name}")  # Adding the title

        # Set fixed limits for x and y axes
        plt.xlim(600, 2000)
        plt.ylim(0, 2200)

        node_degree_legend_elements = [
            plt.scatter([], [], s=5, marker='o', edgecolor='black', facecolor='none', label='5'),
            plt.scatter([], [], s=20, marker='o', edgecolor='black', facecolor='none', label='20'),
            plt.scatter([], [], s=50, marker='o', edgecolor='black', facecolor='none', label='50')
        ]

        node_type_legend_elements = [
            plt.scatter([], [], s=50, marker='o', edgecolor='black', facecolor='grey', label='Broker'),
            plt.scatter([], [], s=50, marker='o', edgecolor='black', facecolor='red', label='Sender'),
            plt.scatter([], [], s=50, marker='o', edgecolor='black', facecolor='blue', label='Receiver')
        ]

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
        
    sttc_neuron_plotter(sub_start, colors,line_threshold)