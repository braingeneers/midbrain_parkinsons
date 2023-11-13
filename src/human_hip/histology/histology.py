#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


def plot_histology(sd, image_path, electrodes=True, neurons=True):
    """
    Function: Plots histology image with electrodes and neurons overlaid.
    Inputs: 
        sd (SpikeData object): contains the metadata necessary to plot the neurons and electrodes
        image_path (string): path to histology image
        electrodes (True/False): Whether or not to plot electrodes
        neurons (True/Fasle): Whether or not to plot neurons
    Outputs:
        Displays a plot of the histology image with electrodes and neurons overlaid
    """
                                                
    # Add background image
    plt.figure(figsize=(15,10))       # Set image size to roughly the shape of the MEA
    img = plt.imread(image_path)      # Load in image
    plt.imshow(img,  extent=[0, 3850, 0, 2100] ) # Plot image, have it correspond to electrode dimensions

    # Plot electrodes
    if electrodes:        # Get electrode positions from metadata, the plot them
        electrode_mapping = pd.DataFrame.from_dict( sd.metadata[0], orient="index", columns=['x','y']  ) 
        plt.scatter( electrode_mapping.x.values, electrode_mapping.y.values, s=4, c='darkorange')

    # plot neurons
    if neurons:            # Get neuron positions from metadata, then plot them
        neuron_x = []
        neuron_y = []
        for key,val in sd.neuron_data[0].items():
            neuron_x.append( val["position"][0] )
            neuron_y.append( val["position"][1] )
        plt.scatter( neuron_x, neuron_y,  c='magenta', alpha=.6, s=50 )  

    #add legend, axises limits, labels,  and title
    legend_elements = [Patch(facecolor="darkorange"), Patch(facecolor="magenta") ]   # Create colors in legend
    plt.legend(legend_elements, ["Electrode","Neuron"])       # Add legend
    plt.xlim(0, 3850)                                       # Set axis limits to that of the MEA
    plt.ylim(0, 2100)
    plt.xlabel('um')                                         # add axises and title
    plt.ylabel('um')
    plt.title(f"Neuron & Electrode Layout")
    plt.show()  

