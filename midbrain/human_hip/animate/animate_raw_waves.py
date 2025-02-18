#!/usr/bin/env python3


from human_hip.raw_data import get_brain_waves
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation


def animate_waves(raw_dict, wave_type, movie_range_ms=None, frame_interval_ms=100,  filename="waves.mp4", size_exp=1.9, size_scale=90 ):

    # check inputs, get movie_range if not provided
    if wave_type not in ["beta", "alpha", "theta", "delta"]:
        raise ValueError(f"wave_type must be one of ['beta', 'alpha', 'theta', 'delta']")
    if movie_range_ms is None:
        if wave_type=="beta": movie_range_ms = range(0, 2000, 1) 
        elif wave_type=="alpha": movie_range_ms = range(0, raw_dict["data"].shape[1], 3)
        elif wave_type=="theta": movie_range_ms = range(0, raw_dict["data"].shape[1], 5)
        elif wave_type=="delta": movie_range_ms = range(0, raw_dict["data"].shape[1], 13)
    if wave_type=="delta": 
        size_scale= size_scale/5  #size_exp= size_exp/2
    video_length = round( len(movie_range_ms)*frame_interval_ms/1000/60 ,3)
    print(f"Making animation of {video_length} minutes")
    if video_length > 10:
        raise ValueError(f"Video length is over 10 minutes. Please shorten movie_range_ms or frame_interval_ms")

    # Create initial plot to feed into animator
    data = get_brain_waves( raw_dict["data"], raw_dict["frame_rate"] )[wave_type]  # get common brain waves
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






# def animate_waves_theta(raw_dict, movie_range_ms=None, frame_interval_ms=100,  filename="theta_waves.mp4", size_exp=1.5, size_scale=100 ):

#     # Create data for animation
#     data = get_brain_waves( raw_dict["data"], raw_dict["frame_rate"] )["theta"]  # get common brain waves
#     movie_range_ms = range(0, raw_dict["data"].shape[1], 5) if movie_range_ms is None else movie_range_ms
#     video_length = round( len(movie_range_ms)*frame_interval_ms/1000/60 ,3)
#     print(f"Making animation of {video_length} minutes")
#     if video_length > 10:
#         raise ValueError(f"Video length is over 10 minutes. Please shorten movie_range_ms or frame_interval_ms")
    
#     # Create initial plot to feed into animator
#     fig = plt.figure( figsize=(12,12) )
#     ax = plt.axes()
#     norm = plt.Normalize( vmin= np.mean(data)-np.std(data), vmax= np.mean(data)+np.std(data) )
#     scatter = ax.scatter( raw_dict['xy'][:,0], raw_dict['xy'][:,1], c=data[:,0] , norm=norm, cmap=cm.coolwarm,
#                           s=(np.abs(data[:,0])**size_exp)*size_scale ,  alpha=.4 , edgecolor='none' )

#     # animation function.  This is called sequentially
#     def animate(i):
#         scatter.set_array(data[:,i])
#         scatter._sizes = (np.abs(data[:,i])**size_exp)*size_scale
#         ax.set_title(f"{i/1000:.3f} seconds")
#         return scatter
#     anim = FuncAnimation(fig, animate, frames=movie_range_ms , interval=frame_interval_ms, blit=False)
#     anim.save( filename )
#     print( f"Saved animation to {filename}" )




# def animate_waves_alpha(raw_dict, movie_range_ms=None, frame_interval_ms=100,  filename="alpha_waves.mp4", size_exp=2.3, size_scale=80 ):

#     # Create data for animation
#     data = get_brain_waves( raw_dict["data"], raw_dict["frame_rate"] )["alpha"]  # get common brain waves
#     movie_range_ms = range(0, raw_dict["data"].shape[1], 3) if movie_range_ms is None else movie_range_ms
#     video_length = round( len(movie_range_ms)*frame_interval_ms/1000/60 ,3)
#     print(f"Making animation of {video_length} minutes")
#     if video_length > 10:
#         raise ValueError(f"Video length is over 10 minutes. Please shorten movie_range_ms or frame_interval_ms")
    
#     # Create initial plot to feed into animator
#     fig = plt.figure( figsize=(12,12) )
#     ax = plt.axes()
#     norm = plt.Normalize( vmin= np.mean(data)-np.std(data), vmax= np.mean(data)+np.std(data) )
#     scatter = ax.scatter( raw_dict['xy'][:,0], raw_dict['xy'][:,1], c=data[:,0] , norm=norm, cmap=cm.coolwarm,
#                           s=(np.abs(data[:,0])**size_exp)*size_scale ,  alpha=.4 , edgecolor='none' )

#     # animation function.  This is called sequentially
#     def animate(i):
#         scatter.set_array(data[:,i])
#         scatter._sizes = (np.abs(data[:,i])**size_exp)*size_scale
#         ax.set_title(f"{i/1000:.3f} seconds")
#         return scatter
#     anim = FuncAnimation(fig, animate, frames=movie_range_ms , interval=frame_interval_ms, blit=False)
#     anim.save( filename )
#     print( f"Saved animation to {filename}" )