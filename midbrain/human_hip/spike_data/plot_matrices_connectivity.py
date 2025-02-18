#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from human_hip.spike_data.analysis import  correlation_matrix


def plot_sttc_matrix( sd, blur=20, sttc_cutoff_high=1.1, plot_color="magma" ):
    """
    Outputs: plots the STTC and Correlation matrices for neurons in a recording
    Input: 
        sd: spike_data object, the standard data type used by braingeneers 
        plot_color: string, the color map to use for the plots. Good colors include: "magma", "viridis", "plasma", "inferno"
    """
    STTC = sd.spike_time_tilings( blur )  # get the spike time tiling matrix
    STTC = np.where(STTC<sttc_cutoff_high, STTC, STTC*0)
                      
    # subplot of STTC 
    plt.imshow(STTC, cmap=plot_color)       # Show the STTC matrix
    plt.title("STTC")         # Set the title, x and y labels
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.colorbar( shrink=0.3) # Add a colorbar to the plot


def plot_correlation_matrix( sd, blur=5, corr_cutoff_high=1.1, plot_color="magma" ):
    """
    Outputs: plots the STTC and Correlation matrices for neurons in a recording
    Input: 
        sd: spike_data object, the standard data type used by braingeneers 
        plot_color: string, the color map to use for the plots. Good colors include: "magma", "viridis", "plasma", "inferno"
    """
    Corr = correlation_matrix(sd, blur)          # get teh correlation matrix
    Corr = np.where(Corr<corr_cutoff_high, Corr, Corr*0)

    # subplot of STTC 
    plt.imshow(Corr, cmap=plot_color)       # Show the correlation matrix
    plt.title("Correlation Matrix")         # Set the title, x and y labels
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.colorbar( shrink=0.3) # Add a colorbar to the plot




def plot_matrices_connectivity( sd, sttc_cutoff_high=1.1, plot_color="magma"):
    """
    Outputs: plots the STTC and Correlation matrices for neurons in a recording
    Input: 
        sd: spike_data object, the standard data type used by braingeneers 
        plot_color: string, the color map to use for the plots. Good colors include: "magma", "viridis", "plasma", "inferno"
    """
    STTC = sd.spike_time_tilings()  # get the spike time tiling matrix
    STTC = np.where(STTC<sttc_cutoff_high, STTC, STTC*0)
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




def plot_sttc_hist(sd, bin_count=40, sttc_cutoff_high=1.1, sttc_delta=20):

    array = sd.spike_time_tilings(sttc_delta)
    array = np.where(array<sttc_cutoff_high, array, array*0)
    flat_array = array.flatten()
    #flat_array += 1
    mean = np.mean(flat_array)
    median = np.median(flat_array)
    std = np.std(flat_array)
    
    #flat_plus_one = flat_array #+ 1
    log_array = np.log(flat_array)
    log_mean = np.mean(log_array)
    log_median = np.median(log_array)
    
    """positive_logs = np.where(flat_array>=0)
    mean_log = np.mean(log_array)
    median_log = np.median(log_array)"""
    """
    log_array = np.log(flat_plus_one)
    mean_log = np.mean(log_array)
    median_log = np.median(log_array)
    """
    
    fig, plts = plt.subplot_mosaic("AB", figsize=(24,8))
    
    #plts["A"].plot(flat_array, y, color="black")
    #plts["A"].hist(flat_array, bins=bin_count)
    plts["A"].hist(flat_array, density=True, bins=bin_count)
    plts["A"].axvline(x=mean, color='blue', linewidth=1, label=f'mean = {mean:.4f}')
    plts["A"].axvline(x=median, color='red', linewidth=1, label=f'Median = {median:.4f}')
    plts["A"].set_xlabel("Value")
    plts["A"].set_ylabel("Density")
    plts["A"].set_title("Original Array")
    plts["A"].legend()
    
    
    plts["B"].hist(log_array, density=True, bins=bin_count)
    plts["B"].axvline(x=log_mean, color='blue', linewidth=1, label=f'mean = {log_mean:.4f}')
    plts["B"].axvline(x=log_median, color='red', linewidth=1, label=f'Median = {log_median:.4f}')
    plts["B"].set_xlabel("Value")
    plts["B"].set_ylabel("Density")
    plts["B"].set_title("Log of Array")
    plts["B"].legend()
    
    