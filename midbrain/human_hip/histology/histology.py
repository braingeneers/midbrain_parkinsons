#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


def plot_histology(sd, image_path, neuron_color="magenta", electrodes=True, neurons=True, xlim=(0, 3850), ylim=(0, 2100), electrode_mapping=None ):
    """
    Function: Plots histology image with electrodes and neurons overlaid.
    Inputs: 
        sd (SpikeData object): contains the metadata necessary to plot the neurons and electrodes
        image_path (string): path to histology image
        electrodes (True/False): Whether or not to plot electrodes
        neurons (True/Fasle): Whether or not to plot neurons
        neurons (True/Fasle): Whether or not to plot neurons
        xlim, ylim (tuple): x and y limits of the plot. Defaults to the dimensions of the MEA
    Outputs:
        Displays a plot of the histology image with electrodes and neurons overlaid
    """
                                                
    # Add background image
    plt.figure(figsize=(15,10))       # Set image size to roughly the shape of the MEA
    img = plt.imread(image_path)      # Load in image
    plt.imshow(img,  extent=[0, 3850, 0, 2100]) # Plot image, have it correspond to electrode dimensions

    # Plot electrodes
    if electrodes:        # Get electrode positions from metadata, the plot them
        if electrode_mapping is None:
            electrode_mapping =  pd.DataFrame.from_dict( sd.metadata[0], orient="index", columns=['x','y']  ) 
        plt.scatter( electrode_mapping.x.values, electrode_mapping.y.values, s=4, c='darkorange')

    # plot neurons
    if neurons:            # Get neuron positions from metadata, then plot them
        neuron_x = []
        neuron_y = []
        for key,val in sd.neuron_data[0].items():
            neuron_x.append( val["position"][0] )
            neuron_y.append( val["position"][1] )
        plt.scatter( neuron_x, neuron_y,  c=neuron_color, alpha=.8, s=50 )  

    #add legend, axises limits, labels,  and title
    legend_elements = [Patch(facecolor="darkorange"), Patch(facecolor=neuron_color) ]   # Create colors in legend
    plt.legend(legend_elements, ["Electrode","Neuron"])       # Add legend
    plt.xlim( xlim[0], xlim[1] )                                       # Set axis limits to that of the MEA
    plt.ylim( ylim[0], ylim[1])
    plt.xlabel('um')                                         # add axises and title
    plt.ylabel('um')
    plt.title(f"Neuron & Electrode Layout")
    plt.show()  




def plot_histology_electrode_map(sd, electrode_mapping, image_path, electrode_color="darkorange", neuron_color="magenta", electrodes=True, neurons=True, xlim=(0, 3850), ylim=(0, 2100) ):
    """
    Function: Plots histology image with electrodes and neurons overlaid.
    Inputs: 
        sd (SpikeData object): contains the metadata necessary to plot the neurons and electrodes
        image_path (string): path to histology image
        electrodes (True/False): Whether or not to plot electrodes
        neurons (True/Fasle): Whether or not to plot neurons
        neurons (True/Fasle): Whether or not to plot neurons
        xlim, ylim (tuple): x and y limits of the plot. Defaults to the dimensions of the MEA
    Outputs:
        Displays a plot of the histology image with electrodes and neurons overlaid
    """
                                                
    # Add background image
    plt.figure(figsize=(15,10))       # Set image size to roughly the shape of the MEA
    img = plt.imread(image_path)      # Load in image
    plt.imshow(img,  extent=[0, 3850, 0, 2100]) # Plot image, have it correspond to electrode dimensions

    # Plot electrodes
    if electrodes:        # Get electrode positions from metadata, the plot them
        plt.scatter( electrode_mapping[:,0], electrode_mapping[:,1], s=4, c=electrode_color)

    # plot neurons
    if neurons:            # Get neuron positions from metadata, then plot them
        neuron_x = []
        neuron_y = []
        for key,val in sd.neuron_data[0].items():
            neuron_x.append( val["position"][0] )
            neuron_y.append( val["position"][1] )
        plt.scatter( neuron_x, neuron_y,  c=neuron_color, alpha=.8, s=50 )  

    #add legend, axises limits, labels,  and title
    if neurons:
        legend_elements = [Patch(facecolor=electrode_color), Patch(facecolor=neuron_color) ]   # Create colors in legend
        plt.legend(legend_elements, ["Electrode","Neuron"])       # Add legend
    else:
        legend_elements = [Patch(facecolor=electrode_color) ]   # Create colors in legend
        plt.legend(legend_elements, ["Electrode"])       # Add legend       
    plt.xlim( xlim[0], xlim[1] )                                       # Set axis limits to that of the MEA
    plt.ylim( ylim[0], ylim[1])
    plt.xlabel('um')                                         # add axises and title
    plt.ylabel('um')
    plt.title(f"Neuron & Electrode Layout")
    plt.show()  


# def plot_histology_electrode_map(sd, electrode_mapping, image_path, neuron_color="magenta", electrodes=True, neurons=True, xlim=(0, 3850), ylim=(0, 2100) ):
#     """
#     Function: Plots histology image with electrodes and neurons overlaid.
#     Inputs: 
#         sd (SpikeData object): contains the metadata necessary to plot the neurons and electrodes
#         image_path (string): path to histology image
#         electrodes (True/False): Whether or not to plot electrodes
#         neurons (True/Fasle): Whether or not to plot neurons
#         neurons (True/Fasle): Whether or not to plot neurons
#         xlim, ylim (tuple): x and y limits of the plot. Defaults to the dimensions of the MEA
#     Outputs:
#         Displays a plot of the histology image with electrodes and neurons overlaid
#     """
                                                
#     # Add background image
#     plt.figure(figsize=(15,10))       # Set image size to roughly the shape of the MEA
#     img = plt.imread(image_path)      # Load in image
#     plt.imshow(img,  extent=[0, 3850, 0, 2100]) # Plot image, have it correspond to electrode dimensions

#     # Plot electrodes
#     if electrodes:        # Get electrode positions from metadata, the plot them
#         plt.scatter( electrode_mapping[:,0], electrode_mapping[:,1], s=4, c='darkorange')

#     # plot neurons
#     if neurons:            # Get neuron positions from metadata, then plot them
#         neuron_x = []
#         neuron_y = []
#         for key,val in sd.neuron_data[0].items():
#             neuron_x.append( val["position"][0] )
#             neuron_y.append( val["position"][1] )
#         plt.scatter( neuron_x, neuron_y,  c=neuron_color, alpha=.8, s=50 )  

#     #add legend, axises limits, labels,  and title
#     legend_elements = [Patch(facecolor="darkorange"), Patch(facecolor=neuron_color) ]   # Create colors in legend
#     plt.legend(legend_elements, ["Electrode","Neuron"])       # Add legend
#     plt.xlim( xlim[0], xlim[1] )                                       # Set axis limits to that of the MEA
#     plt.ylim( ylim[0], ylim[1])
#     plt.xlabel('um')                                         # add axises and title
#     plt.ylabel('um')
#     plt.title(f"Neuron & Electrode Layout")
#     plt.show()  
