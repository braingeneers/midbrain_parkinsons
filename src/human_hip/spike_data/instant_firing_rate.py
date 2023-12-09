#!/usr/bin/env python3
import numpy as np

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