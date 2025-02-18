#!/usr/bin/env python3

from human_hip.spike_data import  latency_times
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow
import numpy as np
from sklearn import preprocessing
import math
import matplotlib.cm as cm



def animate_latencies(sd, pairs, movie_range_ms=None, frame_interval_ms=500, directed=True, directed_backwards=False, filename="latencies.mp4",
                      latency_ms_cutoff_low=1, latency_ms_cutoff_high=15, plot_vector=False, min_dist=0 ):

    # Create plot of neuron positions
    neuron_xy = []
    for neuron in sd.neuron_data[0].values():
        neuron_xy.append( [neuron['position'][0], neuron['position'][1]] )
    neuron_xy = np.array(neuron_xy)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes()                    # get axis element to later create plot
    ax.scatter( neuron_xy[:,0], neuron_xy[:,1], alpha=0.15, c='grey')

    # Create raster of when latencies occur. "Neurons" are now neuron pairs (n1,n2), where n1->n2
    latency_raster = {}
    for pair in pairs:
        if directed:
            latency_raster[ (pair[0],pair[1]) ] =  latency_times( pair[0], pair[1], sd, ms_cutoff_low=latency_ms_cutoff_low, ms_cutoff_high=latency_ms_cutoff_high, positive_only=True )
            if directed_backwards:
                latency_raster[ (pair[1],pair[0]) ] =  latency_times( pair[1], pair[0], sd, ms_cutoff_low=latency_ms_cutoff_low, ms_cutoff_high=latency_ms_cutoff_high, positive_only=True )
        else:
            latency_raster[ (pair[0],pair[1]) ] = latency_times( pair[0], pair[1], sd, ms_cutoff_low=latency_ms_cutoff_low, ms_cutoff_high=latency_ms_cutoff_high, positive_only=False)
    latency_raster = {k:v for k,v in latency_raster.items() if len(v)>0 } # remove empty lists
    print(f"{sum([ len(x) for x in latency_raster.values() ])} latency events occured in total spikedata")
    for key in list(latency_raster.keys()):
        if math.dist( neuron_xy[key[0]] , neuron_xy[key[1]] ) < min_dist:
            del latency_raster[key]
    if min_dist>0: print(f"{sum([ len(x) for x in latency_raster.values() ])} latency events occured above {min_dist} um distance")

    # Create list of neurons that fire at each timepoint
    movie_range_ms = range(0, int(sd.length), 1000) if movie_range_ms is None else movie_range_ms
    video_length = round( len(movie_range_ms)*frame_interval_ms/1000/60 ,3)
    print(f"Making animation of {video_length} minutes")
    if video_length > 10:
        raise ValueError(f"Video length is over 10 minutes. Please shorten movie_range_ms or frame_interval_ms")    
    neurons_by_time = []
    for timepoint in movie_range_ms:                                # loop over time, each second
        firing_neurons = []
        for pair,times in latency_raster.items() :                                   # loop over neurons
            if np.sum( np.abs( times - timepoint ) < movie_range_ms.step/2 ):             # if neuron pair fires least once within  1 second timepoint
                firing_neurons.append( pair )
        neurons_by_time.append( firing_neurons )

    # Create animation
    cmap = cm.get_cmap('hsv')
    def animate(timepoint):
        ax.set_title(f"{(movie_range_ms.start+timepoint*movie_range_ms.step)/1000:.3f} seconds")
        [ patch.remove() for patch in ax.patches ]
        for pair in neurons_by_time[timepoint]:
            start = neuron_xy[ pair[0] ]
            end = neuron_xy[pair[1]] - start
            end = preprocessing.normalize(end) * 75 if plot_vector  else end
            angle= (math.atan2(-(end[1]), end[0]) + np.pi) / (2 * np.pi)
            ax.add_patch( FancyArrow(  start[0], start[1], end[0], end[1], length_includes_head=True, head_width=25,linewidth=1,
                                      color= cmap(angle), alpha=0.7 ) )
    anim = FuncAnimation(fig, animate, frames=range(len(neurons_by_time)), interval=frame_interval_ms, blit=False) 
    anim.save( filename )
    print( f"Saved animation to {filename}" )


