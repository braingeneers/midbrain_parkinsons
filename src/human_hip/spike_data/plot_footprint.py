#!/usr/bin/env python3

import matplotlib.pyplot as plt
import random

def plot_footprint( sd, neuron_id):
    
    # Scatter plot of neurons
    plt.figure(figsize=(8, 8))
    neuron_x = []
    neuron_y = []
    for neuron in sd.neuron_data[0].values():
        neuron_x.append(neuron['position'][0])
        neuron_y.append(neuron['position'][1])
    plt.scatter(neuron_x, neuron_y, alpha=0.10, c='grey')
  
    # Plot neuron geographic location
    for neighbor in  sd.neuron_data[0][neuron_id]['neighbor_positions']:
        plt.scatter( [neighbor[0]+random.random()*13], [neighbor[1]+random.random()*10], alpha=0.40, c='blue')

    plt.xlabel('um')
    plt.ylabel('um')
    plt.title("Directionality plot")
    plt.show()