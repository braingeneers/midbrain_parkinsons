#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrow
from sklearn import preprocessing
import numpy as np
import sklearn

def vector_plot(sd, pairs, lags, arrow_length=75):

    # Dunno why I have to do this, but it fixes a bug
    pairs = pairs     
    lags = lags

    # Plot original scatter
    neuron_x = []
    neuron_y = []
    for neuron in sd.neuron_data[0].values():
        neuron_x.append(neuron['position'][0])
        neuron_y.append(neuron['position'][1])
    plt.figure(figsize=(8, 8))
    plt.scatter(neuron_x, neuron_y, alpha=0.15, c='grey')

    # make pairs are point in same direction
    for i in range(len(pairs)):
        if lags[i]<0:
            pairs[i] = [ pairs[i][1], pairs[i][0] ]

    # Get the x/y locations of the start and en neurons of each pair
    starts = []
    for start in pairs[:,0]:
        starts.append( [ neuron_x[start], neuron_y[start] ] )
    ends = []
    for end in pairs[:,1]:
        ends.append( [ neuron_x[end], neuron_y[end] ] )
    starts = np.array(starts)
    ends = np.array(ends)

    # Get the directions of arrows, then make of of them the same length
    centered = ends-starts
    normalized = preprocessing.normalize(centered) * arrow_length
    
    # Create Arrows
    arrow_color = "red"
    for i in range(len(starts)):
        arrow = FancyArrow( 
                starts[i][0], starts[i][1], normalized[i][0], normalized[i][1], length_includes_head=True, head_width=25,
                linewidth=1, color=arrow_color, alpha=0.7, edgecolor=arrow_color, facecolor=arrow_color )
        plt.gca().add_patch(arrow)



