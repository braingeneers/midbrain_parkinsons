#!/usr/bin/env python3


from human_hip.raw_data import get_brain_waves
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation




def animate_waves_theta(raw_dict, movie_range_ms=None, frame_interval_ms=100,  filename="theta_waves.mp4", size_exp=1.5, size_scale=100 ):

    # Create data for animation
    waves = get_brain_waves( raw_dict["data"], raw_dict["frame_rate"] ) # get common brain waves
    data = waves["theta"] 
    movie_range_ms = range(0, raw_dict["data"].shape[1], 5) if movie_range_ms is None else movie_range_ms
    video_length = round( len(movie_range_ms)*frame_interval_ms/1000/60 ,3)
    print(f"Making animation of {video_length} minutes")
    if video_length > 10:
        raise ValueError(f"Video length is over 10 minutes. Please shorten movie_range_ms or frame_interval_ms")
    
    # Create initial plot to feed into animator
    fig = plt.figure( figsize=(12,12) )
    ax = plt.axes()
    norm = plt.Normalize( vmin= np.mean(data)-np.std(data), vmax= np.mean(data)+np.std(data) )
    scatter = ax.scatter( raw_dict['xy'][:,0], raw_dict['xy'][:,1], c=data[:,0] , norm=norm, cmap=cm.coolwarm,
                          s=(np.abs(data[:,0])**size_exp)*size_scale ,  alpha=.4 , edgecolor='none' )

    # animation function.  This is called sequentially
    def animate(i):
        scatter.set_array(data[:,i])
        scatter._sizes = (np.abs(data[:,i])**size_exp)*size_scale
        ax.set_title(f"{i/1000:.3f} seconds")
        return scatter
    anim = FuncAnimation(fig, animate, frames=movie_range_ms , interval=frame_interval_ms, blit=False)
    anim.save( filename )
    print( f"Saved animation to {filename}" )



