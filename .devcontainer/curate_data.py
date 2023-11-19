#!/usr/bin/env python3

# import stuff
import numpy as np
import pickle
from braingeneers.analysis.analysis import SpikeData
from human_hip.basics import read_phy_files


#######################################
#######################################
#.         Curate ?Anterior? CA1 Data
#######################################
#######################################

# We load in the ?Anterior? CA1 dataset.
sd = read_phy_files('/workspaces/human_hippocampus/data/ephys/2023-04-02-e-hc328_unperturbed/derived/kilosort2/hc3.28_hckcr1_chip16835_plated34.2_rec4.2_curated.zip')

# Create sub-datasets and get their spike trains
sd_1 = sd.subtime(0, 22.5*1000)
sd_2 = sd.subtime(27.5*1000, 32*1000)
sd_3 = sd.subtime(37.5*1000, 207.5*1000)
sd_4 = sd.subtime(210*1000, sd.length)

spike_train_1 = sd_1.train
spike_train_2 = sd_2.train
spike_train_3 = sd_3.train
spike_train_4 = sd_4.train

#Edit the spike trains so that the timing of spikes are correct
modified_spike_train_2 = [neuron_spike_times + 22.5*1000 for neuron_spike_times in spike_train_2]
modified_spike_train_3 = [neuron_spike_times + 27*1000 for neuron_spike_times in spike_train_3]
modified_spike_train_4 = [neuron_spike_times + 197*1000 for neuron_spike_times in spike_train_4]

# function to Combine spike trains into a singe train
def combine_multiple_spike_trains(spike_train_list):
    num_neurons = len(spike_train_list[0])
    num_spike_trains = len(spike_train_list)
    combined_spike_train = [np.array([]) for _ in range(num_neurons)]
    
    for neuron in range(num_neurons):
        combined_neuron = np.array([])  # Initialize an empty array for each neuron
        for i in range(num_spike_trains):
            combined_neuron = np.append(combined_neuron, spike_train_list[i][neuron])
        combined_spike_train[neuron] = combined_neuron
    
    return combined_spike_train

# Call the function to combine spike trains
combined_spike_train = combine_multiple_spike_trains([spike_train_1, modified_spike_train_2, modified_spike_train_3, modified_spike_train_4])
#print("Combined Spike Train:", len(combined_spike_train[0]))

#turn spike train into a `spike_data` object
upd_sd2 = SpikeData(combined_spike_train, length=(sd_1.length + sd_2.length + sd_3.length + sd_4.length), N=sd_1.N, 
                         metadata=sd_1.metadata, neuron_data=sd_1.neuron_data,
                         neuron_attributes=sd_1.neuron_attributes)

# Save Dataset
with open( '/workspaces/human_hippocampus/data/ephys/2023-04-02-e-hc328_unperturbed/StitchedDataUpdated.pkl' , 'wb') as file:
    pickle.dump(upd_sd2, file)



#######################################
#######################################
#.        Curate ?Posterior? Data
#######################################
#######################################

#We load in the unknown dataset.
sd2 = read_phy_files('/workspaces/human_hippocampus/data/ephys/2023-05-10-e-hc52_18790_unperturbed/derived/kilosort2/hc5.2_chip18790_baseline_rec5.10.23_curated_s1.zip')

#The data look mostly reasonable, except for the very end of the recording, so we remove only this portion of the data
new_cut_sd2 = sd2.subtime(0, 106.5*1000)

# Save data
with open( '/workspaces/human_hippocampus/data/ephys/2023-05-10-e-hc52_18790_unperturbed/5-10-23s1-CutData.pkl' , 'wb') as file:
    pickle.dump(new_cut_sd2, file)


