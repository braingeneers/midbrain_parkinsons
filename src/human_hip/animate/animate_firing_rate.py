

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from human_hip.spike_data import firing_rates
import numpy as np



def animate_firing_rate(sd, movie_range_ms=None, frame_interval_ms=100,  filename="firing_rate.mp4", size_exp=1.4, size_scale=3 ):

    # creat movie range
    movie_range_ms = range(0, int(sd.length), 1000) if movie_range_ms is None else movie_range_ms
    video_length = round( len(movie_range_ms)*frame_interval_ms/1000/60 ,3)
    print(f"Making animation of {video_length} minutes")

    # get neuron positions
    neuron_xy = []
    for neuron in sd.neuron_data[0].values():
        neuron_xy.append( [neuron['position'][0], neuron['position'][1]] )
    neuron_xy = np.array(neuron_xy)

    # get axis element and create plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes()  
    scatter = ax.scatter( neuron_xy[:,0], neuron_xy[:,1], s=(firing_rates(sd)**size_exp)*size_scale, alpha=.4,  c='red')

    # animation function.  This is called sequentially
    def animate(i):
        sd_small = sd.subtime( i, i+movie_range_ms.step )
        scatter._sizes = (firing_rates(sd_small)**size_exp) * size_scale
        ax.set_title(f"{i/1000:.3f} seconds")
        return scatter
    anim = FuncAnimation(fig, animate, frames=movie_range_ms , interval=frame_interval_ms, blit=False)
    anim.save( filename )
    print( f"Saved animation to {filename}" )





















