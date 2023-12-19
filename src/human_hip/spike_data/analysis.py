#!/usr/bin/env python3

from scipy.ndimage import gaussian_filter1d
import numpy as np



def firing_rates(spike_data):
    """
    input: a spike data object, the common data format used by braingeneers
    output: a numpy array of the mean firing rate of each neuron in the spike data object
    """
    mean_firing_rates = []
    for neuron_spikes in spike_data.train:
        num_spikes = len(neuron_spikes)
        time_duration = spike_data.length / 1000  # Assuming spike times are in milliseconds
        firing_rate = num_spikes / time_duration
        mean_firing_rates.append(firing_rate)
    return np.array(mean_firing_rates)




def ISI(sd, neuron=-1, max_isi=100):
    """
    inputs:
        sd: a spike data object, the common data format used by braingeneers
        neuron: the neuron to get the interspike intervals for, if -1, all neurons are used
        max_isi: the maximum allowed time (ms) between spikes, ISI's above this value are not included
    output: a list of all the interspike intervals for the specified neuron, or all neurons if neuron=-1
    """
    if neuron == -1:
        isis_raw = sd.interspike_intervals()
        isis=[]
        for isis_neuron in isis_raw:   
            isis = isis + isis_neuron[isis_neuron<max_isi].tolist() 
    else:
        isis_neuron = sd.interspike_intervals()[neuron]
        isis = isis_neuron[isis_neuron<max_isi].tolist() 
    return isis




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




def eigenvalues_eigenvectors(A):
    W, U = np.linalg.eigh(A)
    # The rank of A can be no greater than the smaller of its
    # dimensions, so cut off the returned values there.
    rank = min(*A.shape)
    U = U[:,-rank:]
    sgn = (-1)**(U[0,:] < 0)
    # Also reverse the order of the eigenvalues because eigh()
    # returns them in ascending order but descending makes more sense.
    return W[-rank:][::-1], (U*sgn[np.newaxis,:])[:, ::-1]




def instant_firing_rate(sd, neuron_num, max_ifr=9e10 ):
    """
    Function: Calculates instant firing rate for a single neuron (the inverse of the ISI)
    Inputs:
        sd: SpikeData object
        neuron_num (int): index of neuron to calculate instant firing rate of
        max_ifr (int): maximum possible value of instant firing rate. Stops the rate from being too high
    Outputs:
        instant_fire_rate (list):  list of the instant firing rate at every ms in time
    """
    spike_times= np.unique( np.round(sd.train[neuron_num]).astype(int) )  # get spike times, round to nearest ms, remove duplicates
    instant_fire_rate = []
    last_spike = 0
    for spike_time in spike_times:          # for every spike
        isi = spike_time-last_spike         # calculate time in between spikes  
        instant_fire_rate+= [max_ifr]*isi if 1/isi>max_ifr else [1/isi]*isi   # calculate instant firing rate, add to list
        last_spike = spike_time             # update last spike time
    return instant_fire_rate
















