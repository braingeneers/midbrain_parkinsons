#!/usr/bin/env python3

#### Summary ####
# This code contains the basic plots used in the first round f analysis of the data
# Thes plots include a spike raster, .... (add more later)

import matplotlib.pyplot as plt
from braingeneers.analysis.analysis import SpikeData
from scipy.ndimage import gaussian_filter1d
import numpy as np

# Plots a simple spike raster
def raster_plot(sd, ax):
    """"
    param sd: a spike data object, the common data format used by braingeneers
    param ax: the axis object used by matplotlib, it can be called later for ploting
    """
    idces, times = sd.idces_times()
    ax.scatter(times/1000, idces, marker='|', s=1)
    ax.set_xlabel("Time(s)")
    ax.set_ylabel('Unit #')
    ax.set_title("Raster Plot")
    

# Fancier raster plot, where you can specify sub-times to plot, and the plot size
def raster_fancy_plot(sd, xsize=10, ysize=6, start_time=0, stop_time=None, save_path=None):
    """"
    param sd: a spike data object, the common data format used by braingeneers
    params xsize, ysize: the size of the x/y axises for the resulting plot
    params start_time, stop_time: the start/stop times of the recording which we will plot (in seconds)
    param save_path: where to save the resulting plot. If None, plot is presented to screen
    return: this function implicitly returns a plot to a notebook (through show) or it saves the plot
    """
    # Zoomed Raster and pop rate
    # Get coordinates for raster
    idces, times = sd.idces_times()

    # Get population rate for everything
    pop_rate = sd.binned(bin_size=1)  # in ms
    # Lets smooth this to make it neater
    sigma = 5
    pop_rate_smooth = gaussian_filter1d(pop_rate.astype(float), sigma=sigma)
    t = np.linspace(0, sd.length, pop_rate.shape[0]) / 1000

    # Determine the stop_time if it's not provided
    if stop_time is None:
        stop_time = t[-1]

    # Filter times and idces within the specified start and stop times
    mask = (times >= start_time * 1000) & (times <= stop_time * 1000)
    times = times[mask]
    idces = idces[mask]

    fig, ax = plt.subplots(figsize=(xsize, ysize))

    ax.scatter(times / 1000, idces, marker='|', s=1)
    ax2 = ax.twinx()
    ax2.plot(t, pop_rate_smooth, c='r')

    ax.set_xlim(start_time, stop_time)
    ax.set_xlabel("Time(s)")
    ax.set_ylabel('Unit #')
    ax2.set_ylabel('Firing Rate')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()