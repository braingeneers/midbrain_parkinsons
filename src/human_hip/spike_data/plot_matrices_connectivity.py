#!/usr/bin/env python3

from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt


def correlation_matrix(sd, blur=5):
    """
    Output: returns the correlation matrix for neurons in a recording
    Inputs:
        sd: spike_data object, the standard data type used by braingeneers
        blur: the sigma value for the gaussian filter used to smooth the raster
    """
    dense_raster = sd.raster(bin_size=1)  # create a spike raster with each column being 1ms, and each row being a neuron
    blurred_raster = gaussian_filter1d(dense_raster.astype(float),sigma=blur) # smooth the raster
    return np.corrcoef( blurred_raster ) # return the correlation matrix


def plot_sttc_matrix( sd, blur=20 ):
    """
    Outputs: plots the STTC and Correlation matrices for neurons in a recording
    Input: 
        sd: spike_data object, the standard data type used by braingeneers 
    """
    STTC = sd.spike_time_tilings( blur )  # get the spike time tiling matrix
                      
    # subplot of STTC 
    plt.imshow(STTC)       # Show the STTC matrix
    plt.title("STTC")         # Set the title, x and y labels
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.colorbar( shrink=0.3) # Add a colorbar to the plot


def plot_correlation_matrix( sd, blur=5 ):
    """
    Outputs: plots the STTC and Correlation matrices for neurons in a recording
    Input: 
        sd: spike_data object, the standard data type used by braingeneers 
    """
    Corr = correlation_matrix(sd, blur)          # get teh correlation matrix
                      
    # subplot of STTC 
    plt.imshow(Corr)       # Show the correlation matrix
    plt.title("Correlation Matrix")         # Set the title, x and y labels
    plt.xlabel("Neuron")
    plt.ylabel("Neuron")
    plt.colorbar( shrink=0.3) # Add a colorbar to the plot


def plot_matrices_connectivty(sd):
    """
    Outputs: plots the STTC and Correlation matrices for neurons in a recording
    Input: 
        sd: spike_data object, the standard data type used by braingeneers 
    """
    STTC = sd.spike_time_tilings()  # get the spike time tiling matrix
    Corr = correlation_matrix(sd)          # get teh correlation matrix
                  
    fig, plots = plt.subplot_mosaic( """AB"""  , figsize=(12,10))   # Set up layout for 2 figures,
    
    # subplot of STTC 
    pltA = plots["A"].imshow(STTC)       # Show the STTC matrix
    plots["A"].set_title("STTC")         # Set the title, x and y labels
    plots["A"].set_xlabel("Neuron")
    plots["A"].set_ylabel("Neuron")
    fig.colorbar(pltA, ax=plots["A"], shrink=0.3) # Add a colorbar to the plot
    
    # subplot of Correlation
    pltB = plots["B"].imshow(Corr)      # Show the correlation matrix
    plots["B"].set_title("Correlation") # Set the title, x and y labels
    plots["B"].set_xlabel("Neuron")
    plots["B"].set_ylabel("Neuron")
    fig.colorbar(pltB, ax=plots["B"], shrink=0.3) # Add a colorbar to the plot



