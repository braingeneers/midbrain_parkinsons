#!/usr/bin/env python3  



import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import pickle
import braingeneers.data.datasets_electrophysiology as ephys



def data_get_experiments(uuid):
    """
    Input uuid: the uuid of the experiment
    Output: Prints the filename corresponding to each experiment
    """
    metadata = ephys.load_metadata(uuid)
    for key,val in metadata["ephys_experiments"].items():
        print( key," - ", val["blocks"][0]["path"] )


def data_create(uuid, experiment_name, start_s, save_path, length_s=10):
    """
    Input:
        uuid: the uuid of the experiment
        experiment_name: the name of the experiment corresponding to the file on interest
        start_s: the start time of the recording in seconds
        save_path: the path to save for the data (must be a pickle file)
        length_s: the length of the resulting recording data in seconds 
    Output: Saves the data to save_path
    """
    print("Loading Data... might take up to 10min")
    metadata = ephys.load_metadata(uuid)
    raw_data = ephys.load_data( metadata=metadata, experiment=experiment_name, offset=start_s*20000, length=length_s*20000, channels=None )
    channel_map = np.array( metadata['ephys_experiments'][experiment_name]["mapping"] )

    data_down = []  # the variable that will hold the downsambled data
    for i in channel_map[:,0].astype(int) : # we gather data for ever channel that was recorded from, (these channels are in the channel map of the metadata)
        data_down.append( signal.decimate( raw_data[i,:], 20 )  ) # we get everyt 20th data point, andthen append it to the data_down variable
    del raw_data
    data_down = np.array( data_down ) # we turn the data into an np.array for easier future analysis

    print("Saving Data...")
    to_pickle = {"data": data_down, "xy": channel_map[:,1:3], "frame_rate": 20000/20, "uuid":uuid,
                  "file": metadata["ephys_experiments"][experiment_name]["blocks"][0]["path"] }
    del data_down
    with open( save_path, 'wb') as filename:
        pickle.dump( to_pickle, filename)
    print("Done!")



