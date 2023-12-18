#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from human_hip.spike_data.analysis import correlation_matrix


def plot_sttc_matrix( sd, blur=20, plot_color="magma" ):
    """
    Outputs: plots the STTC and Correlation matrices for neurons in a recording
    Input: 
        sd: spike_data object, the standard data type used by braingeneers 
        plot_color: string, the color map to use for the plots. Good colors include: "magma", "viridis", "plasma", "inferno"
    """
    STTC = sd.spike_time_tilings( blur )  # get the spike time tiling matrix
                      
    # subplot of STTC 
    plt.imshow(STTC, cmap=plot_color)       # Show the STTC matrix
    plt.title("STTC")         # Set the title, x and y labels
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.colorbar( shrink=0.3) # Add a colorbar to the plot


def plot_correlation_matrix( sd, blur=5, plot_color="magma" ):
    """
    Outputs: plots the STTC and Correlation matrices for neurons in a recording
    Input: 
        sd: spike_data object, the standard data type used by braingeneers 
        plot_color: string, the color map to use for the plots. Good colors include: "magma", "viridis", "plasma", "inferno"
    """
    Corr = correlation_matrix(sd, blur)          # get teh correlation matrix
                      
    # subplot of STTC 
    plt.imshow(Corr, cmap=plot_color)       # Show the correlation matrix
    plt.title("Correlation Matrix")         # Set the title, x and y labels
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.colorbar( shrink=0.3) # Add a colorbar to the plot


def plot_matrices_connectivity( sd, plot_color="magma"):
    """
    Outputs: plots the STTC and Correlation matrices for neurons in a recording
    Input: 
        sd: spike_data object, the standard data type used by braingeneers 
        plot_color: string, the color map to use for the plots. Good colors include: "magma", "viridis", "plasma", "inferno"
    """
    STTC = sd.spike_time_tilings()  # get the spike time tiling matrix
    Corr = correlation_matrix(sd)          # get teh correlation matrix
    fig, plots = plt.subplot_mosaic( """AB"""  , figsize=(14,7) )   # Set up layout for 2 figures,
    
    # subplot of STTC 
    pltA = plots["A"].imshow(STTC, cmap=plot_color)       # Show the STTC matrix
    plots["A"].set_title("STTC")         # Set the title, x and y labels
    plots["A"].set_xlabel("Neuron")
    plots["A"].set_ylabel("Neuron")
    fig.colorbar(pltA, ax=plots["A"], shrink=0.5) # Add a colorbar to the plot
    
    # subplot of Correlation
    pltB = plots["B"].imshow(Corr, cmap=plot_color)      # Show the correlation matrix
    plots["B"].set_title("Correlation") # Set the title, x and y labels
    plots["B"].set_xlabel("Neuron")
    plots["B"].set_ylabel("Neuron")
    fig.colorbar(pltB, ax=plots["B"], shrink=0.5) # Add a colorbar to the plot

