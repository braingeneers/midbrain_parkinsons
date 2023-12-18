#!/usr/bin/env python3

from scipy.ndimage import gaussian_filter1d
import numpy as np


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





















