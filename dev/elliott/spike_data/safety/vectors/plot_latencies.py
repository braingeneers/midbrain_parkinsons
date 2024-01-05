#!/usr/bin/env python3

import matplotlib.pyplot as plt


def plotLagsLayout(sd, lags, pairs):
    
    neuron_x = []
    neuron_y = []
    for neuron in sd.neuron_data[0].values():
        neuron_x.append(neuron['position'][0])
        neuron_y.append(neuron['position'][1])
    plt.figure(figsize=(8, 8))
    plt.scatter(neuron_x, neuron_y, alpha=0.15, c='grey')

    for i in range(len(lags)):
        if lags[i]<0:
            start_i = pairs[i][0] 
            end_i = pairs[i][1] 
        else:
            start_i = pairs[i][1] 
            end_i = pairs[i][0] 

        arrow_color = "red"
        arrow = FancyArrow(
            neuron_x[end_i], neuron_y[end_i],
            neuron_x[start_i] - neuron_x[end_i], neuron_y[start_i] - neuron_y[end_i],
            length_includes_head=True, head_width=25,
            linewidth=1, color=arrow_color, alpha=0.7, edgecolor=arrow_color, facecolor=arrow_color)
        plt.gca().add_patch(arrow)

    plt.xlabel('um')
    plt.ylabel('um')
    plt.title("Cross Correlation Between 2-10ms")
    plt.show()