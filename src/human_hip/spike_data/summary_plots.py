#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d




### Replaced this code with a built in braingeneers function in the spike data object
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




def summary_plots(sd):
    """
    input: a spike data object, the common data format used by braingeneers
    output: plots for ISI histogram, firing rate (histogram & layout), and Spikeraster of first 30 seconds
    """
    my_firing_rates = firing_rates(sd)  #my_firing_rates = sd.rates()
    seconds=30 # seconds to display raster
    neuron_x = []
    neuron_y = []
    
    for neuron in sd.neuron_data[0].values():
        neuron_x.append(neuron["position"][0])
        neuron_y.append(neuron["position"][1])
    
    # Plot main figure --------------------------------------------------------------------
    figs, plots = plt.subplots(nrows=2,ncols=2,figsize=(12,12))
    figs.suptitle(f"Plots of recording:", ha="center")
    
    # Plot ISI Histogram subplot
    plots[0,0].hist(ISI(sd), bins=50);
    plots[0,0].set_title("Interspike Interval of Recording")
    plots[0,0].set_xlabel("Time bin(ms)")
    plots[0,0].set_ylabel("ISI count")
    
    # Plot Firing Rates Histogram subplot
    plots[0,1].hist(my_firing_rates, bins=50)
    plots[0,1].set_title("Average Firing Rate for Neural Units") 
    plots[0,1].set_xlabel('Firing rate, Hz')
    plots[0,1].set_ylabel('Number neural units') 


    # Plot Neuron Firing Rate Layout subplot
    #plots[1,0].scatter(neuron_x, neuron_y, s=(2**my_firing_rates)*100, c="red", alpha=0.3)
    plots[1,0].scatter(neuron_x, neuron_y, s=(2**my_firing_rates)*10, c="red", alpha=0.3)
    plots[1,0].set_title("Neuron Firing Rate Across MEA")
    plots[1,0].set_xlabel("um")
    plots[1,0].set_ylabel("um")
    #plots[3] = Firing_Rate_Layout(sd);
    
    
    # Plot Raster with plotted firing rate over time subplot
    # Zoomed Raster and pop rate
    # Get coordinates for raster
    idces, times = sd.idces_times()
    
    # Get population rate for everything
    pop_rate = sd.binned(bin_size=1)# in ms
    # Lets smooth this to make it neater
    sigma = 5
    pop_rate_smooth = gaussian_filter1d(pop_rate.astype(float),sigma=sigma) 
    t = np.linspace(0,sd.length,pop_rate.shape[0])/1000
    
    plots[1,1].scatter(times/1000,idces,marker='|',s=1)
    plots2 = plots[1,1].twinx()
    plots2.plot(t,pop_rate_smooth,c='r')

    plots[1,1].set_xlim(0,seconds)
    plots[1,1].set_title("Spike Raster Analysis")
    plots[1,1].set_xlabel("Time(s)")
    plots[1,1].set_ylabel("Unit #")
    plots2.set_ylabel("Firing Rate")
    
    # Plot second figure ------------------------------------------------------------------
    figs2, axs = plt.subplots(nrows=2,ncols=4,figsize=(30,10)) 
    figs2.suptitle(f"Interspike Interval of Individual Neural Units")
    
    for i in range(8): # Plot individual ISI figures
        if(i < sd.N):
            if i < 4: # First Row
                axs[0,i].hist(ISI(sd, neuron=i))
                axs[0,i].set_title(f"Interspike Interval of Neural Unit {i}")
                axs[0,i].set_xlabel("Time bin(ms)")
                axs[0,i].set_ylabel("ISI count")
            else: # Second Row
                axs[1,i-4].hist(ISI(sd, neuron=i))
                axs[1,i-4].set_title(f"Interspike Interval of Neural Unit {i}")
                axs[1,i-4].set_xlabel("Time bin(ms)")
                axs[1,i-4].set_ylabel("ISI count")
        else: # Print warning title in case neuron count is uner 8
            figs2.suptitle(f"Interspike Interval of Individual Neural Units \n Note: Neuron Count Under 8 ({sd.N})")


