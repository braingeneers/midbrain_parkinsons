#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrow
from sklearn import preprocessing
import numpy as np
from human_hip.spike_data import latencies, latency_times, plot_raster, plot_footprint
from braingeneers.analysis.analysis import SpikeData
import warnings
import diptest 



### Note: this code isn't done
# It should state whether or not to plot directed or underected latencies 
# It should also pass all of the parameters used in plot_raster
def plot_raster_latency_pairs(sd, pairs):
    latency_raster = []
    for pair in pairs:
        latency_raster.append( latency_times( pair[0], pair[1], sd, ms_cutoff=15, positive_only=False) )
    sd_latency = SpikeData(latency_raster)
    plot_raster( sd_latency )


# The function creates  plot of arrows show the direction that information is flowing out of neurons
def plot_vector_layout( sd, pairs, normalize=True, arrow_length=75):
    """
    Inputs:
        pairs: np.array of neuron indices (as pairs) for which a connection exists, ex: [[0,1], [0,2], [2,3]]
        lags: np.array of the average lag time in ms corresponding to the neuron pairs, ex: [1, 3, -4]
        normalize: boolean, if True, all arrows will be the same length, if False, arrows will point to the ending neuron
        arrow_length: integer of how long the arrows should be drawn on the final plot
    Outputs:
        A plot depicting th 2D locations of neurons, with arrows showing the direction of information flow
    """
    # Get the x/y locations of the start and end neurons of each pair
    neuron_xy = []
    for neuron in sd.neuron_data[0].values():
        neuron_xy.append( [neuron['position'][0], neuron['position'][1]] )
    neuron_xy = np.array(neuron_xy)

    # Plot original scatter
    plt.figure(figsize=(8, 8))
    plt.scatter( neuron_xy[:,0], neuron_xy[:,1], alpha=0.15, c='grey')

    # make pairs point in same direction
    pairs = pairs                         # make a copy of pairs, this avoids some bug
    for i in range(len(pairs)):
        lag = np.median(latencies( pairs[i][0], pairs[i][1], sd, ms_cutoff=20))
        if lag<0:
            pairs[i] = [ pairs[i][1], pairs[i][0] ]

    # Creat arrows show angle of information flow from a neuron
    starts = neuron_xy[ pairs[:,0] ]  # Get the x/y locations of the start and end neurons of each pair
    ends = neuron_xy[ pairs[:,1] ]
    centered = ends-starts   # Get the directions of arrows, then make of of them the same length
    normalized = preprocessing.normalize(centered) * arrow_length if normalize else centered # make same lengths, unless told otherwise
    
    # Draw Arrows
    arrow_color = "red"
    for i in range(len(starts)):
        arrow = FancyArrow( 
                starts[i][0], starts[i][1], normalized[i][0], normalized[i][1], length_includes_head=True, head_width=25,
                linewidth=1, color=arrow_color, alpha=0.7, edgecolor=arrow_color, facecolor=arrow_color )
        plt.gca().add_patch(arrow)








def plot_pair_analysis( n1, n2, sd):
    
    lag = np.median(latencies( n1, n2, sd, ms_cutoff=20))
    if lag<0:
        start_i = n2
        end_i = n1
    else:
        start_i = n1
        end_i = n2

    # Layout Plot
    warnings.filterwarnings("ignore")
    plot_footprint( sd, [start_i, end_i] )
    
    # Summary Stats
    latencies_raw = latencies( start_i, end_i, sd  )
    latencies_clean = latencies_raw[ np.where( np.abs(latencies_raw) < 15 )[0] ]
    print( "Number of Latencies", len(latencies_clean) )
    print( "Mean Latency", round(np.mean(latencies_clean), 3) )
    print( "Median Latency", round(np.median(latencies_clean), 3) )
    #print( "Latency Probability", {round(100*(len(lates_filtered)/len(lates_raw)))})
    print("STTC", sd.spike_time_tiling( start_i, end_i) )
    print("Diptest P-val", round( diptest.diptest(latencies_clean)[1] , 3) )
    print( "Latency Probability", round(len(latencies_clean)/len(latencies_raw), 3) )
    print(".")
    print( "Sender Neuron", start_i)
    print( "Sender   Firing Rate", round(sd.rates(unit='Hz')[start_i] , 3) )
    print(".")
    print("Receiver Neuron", end_i)
    print( "Receiver Firing Rate", round(sd.rates(unit='Hz')[end_i], 3) )
    
    # Plot other graphs
    figs, plots = plt.subplots(nrows=3,ncols=2,figsize=(14,10))

    mean_latency = np.mean(latencies_clean)     # Get mean and SD
    std_latency = np.std(latencies_clean)
    cutoff = 2 * std_latency           # remove outliers
    lates_filtered = [latency for latency in latencies_clean if abs(latency - mean_latency) <= cutoff]
    plots[0,0].hist(lates_filtered, bins=12, alpha=0.7, label='Latency')
    plots[0,0].axvline(mean_latency, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plots[0,0].axvline(mean_latency - std_latency, color='green', linestyle='dashed', linewidth=2, label='Std -')
    plots[0,0].axvline(mean_latency + std_latency, color='green', linestyle='dashed', linewidth=2, label='Std +')
    plots[0,0].axvline(0, color='black', linestyle='dashed', linewidth=0.5, label='Std +')
    plots[0,0].set_xlim(-1*(abs(mean_latency)+cutoff), abs(mean_latency) + cutoff) 
    plots[0,0].set_xlabel("Latency (ms)")
    plots[0,0].set_ylabel("Count")
    plots[0,0].set_title(f"Fancy Latency Histogram")
    plots[0,0].legend()

    plots[0,1].hist(latencies_clean, bins=12)
    plots[0,1].set_title("Latency Histogram")
    plots[0,1].set_xlabel("Latency (ms)")
    plots[0,1].set_ylabel("Count")

    plots[1,0].hist(sd.interspike_intervals()[start_i], bins=50);
    plots[1,0].set_title("Sender ISI")
    plots[1,0].set_xlabel("Time bin(ms)")
    plots[1,0].set_ylabel("ISI count")

    plots[1,1].hist(sd.interspike_intervals()[end_i], bins=50);
    plots[1,1].set_title("Receiver ISI")
    plots[1,1].set_xlabel("Time bin(ms)")
    plots[1,1].set_ylabel("ISI count")
    
    
    plots[2,0].plot( sd.neuron_data[0][start_i]["template"] )
    plots[2,0].set_title("Sender Spike Waveform")
    plots[2,0].set_xlabel("")
    plots[2,0].set_ylabel("")
    
    plots[2,1].plot( sd.neuron_data[0][end_i]["template"] )
    plots[2,1].set_title("Receiver Spike Waveform")
    plots[2,1].set_xlabel("")
    plots[2,1].set_ylabel("")







##################
#      OLD
##################

# # The function creates  plot of arrows show the direction that information is flowing out of neurons
# def plot_vector_layout( pairs, lags, neuron_xy, normalize=True, arrow_length=75):
#     """
#     Inputs:
#         pairs: np.array of neuron indices (as pairs) for which a connection exists, ex: [[0,1], [0,2], [2,3]]
#         lags: np.array of the average lag time in ms corresponding to the neuron pairs, ex: [1, 3, -4]
#         neuron_xy: np.array of the x/y locations of the neurons, ex: [ [0,23.5], [13,35], [56,24] ]
#         normalize: boolean, if True, all arrows will be the same length, if False, arrows will point to the ending neuron
#         arrow_length: integer of how long the arrows should be drawn on the final plot
#     Outputs:
#         A plot depicting th 2D locations of neurons, with arrows showing the direction of information flow
#     """
#     # Plot original scatter
#     plt.figure(figsize=(8, 8))
#     plt.scatter( neuron_xy[:,0], neuron_xy[:,1], alpha=0.15, c='grey')

#     # make pairs point in same direction
#     pairs = pairs                         # make a copy of pairs and lags, this avoids some bug
#     lags = lags
#     for i in range(len(pairs)):
#         if lags[i]<0:
#             pairs[i] = [ pairs[i][1], pairs[i][0] ]

#     # Creat arrows show angle of information flow from a neuron
#     starts = neuron_xy[ pairs[:,0] ]  # Get the x/y locations of the start and end neurons of each pair
#     ends = neuron_xy[ pairs[:,1] ]
#     centered = ends-starts   # Get the directions of arrows, then make of of them the same length
#     normalized = preprocessing.normalize(centered) * arrow_length if normalize else centered # make same lengths, unless told otherwise
    
#     # Draw Arrows
#     arrow_color = "red"
#     for i in range(len(starts)):
#         arrow = FancyArrow( 
#                 starts[i][0], starts[i][1], normalized[i][0], normalized[i][1], length_includes_head=True, head_width=25,
#                 linewidth=1, color=arrow_color, alpha=0.7, edgecolor=arrow_color, facecolor=arrow_color )
#         plt.gca().add_patch(arrow)




