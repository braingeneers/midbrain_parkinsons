#!/usr/bin/env python3

import numpy as np




def getLatencies(neuron1, neuron2, sd, ms_cutoff=9e12):
    """
    Function:
        returns all the latencies that occur between two neurons, n1 and n2. 
        A latency is defined as the time difference between a spike from n1 and the nearest spike from n2 (positive or negative)
    Inputs:
        neuron1 (integer): index of the neuron, n1 
        neuron2 (integer): index of the neuron, n2, *note* if n1 occurs before n2, the latency will be positive
        sd (SpikeData object): contains the original spike data from the recording
        ms_cutoff (integer): the maximum latency to be considered, in milliseconds
    Outputs:
        cur_latencies (np.array): the time difference between n1 and n2, for ever latency occurence less than ms_cutoff
    """
    train1 = sd.train[ neuron1 ]
    train2 = sd.train[ neuron2 ]    
    cur_latencies = []
    for time in train1:
        abs_diff_ind = np.argmin(np.abs(train2 - time))  # Subtract time from all spikes in the train and take the absolute value        
        latency = np.array(train2)-time       # Calculate the actual latency
        latency = latency[abs_diff_ind]

        if np.abs(latency) <= ms_cutoff:     # Only append latencies that are within a certain time cutoff
            cur_latencies.append(latency)
    return np.array(cur_latencies)



def getLatencyTimes(neuron1, neuron2, sd, ms_cutoff=9e12, positive_only=False):
    """
    Function:
        returns the timepoints (in ms) for which a latency occurs between two neurons, n1 and n2. 
        A latency is defined as the time difference between a spike from n1 and the nearest spike from n2 (positive or negative)
    Inputs:
        neuron1 (integer): index of the neuron, n1 
        neuron2 (integer): index of the neuron, n2, *note* if n1 occurs before n2, the latency will be positive
        sd (SpikeData object): contains the original spike data from the recording
        ms_cutoff (integer): the maximum latency to be considered, in milliseconds
        positive_only (boolean): if True, only return latencies for which n1 occurs before n2
    Outputs:
        cur_latencies (np.array): the timepoints for which every latency occurs between n1 and n2, that is less than ms_cutoff
    """
    train1 = sd.train[ neuron1 ]
    train2 = sd.train[ neuron2 ]    
    cur_latencies = []
    for time in train1:
        abs_diff_ind = np.argmin(np.abs(train2 - time))  # Subtract time from all spikes in the train and take the absolute value        
        latency = np.array(train2)-time       # Calculate the actual latency
        latency = latency[abs_diff_ind]
        #time2 = train2[abs_diff_ind]#print(time, time2, latency)

        if np.abs(latency) <= ms_cutoff:     # Only append latencies that are within a certain time cutoff
            if positive_only:
                if latency>0:
                    cur_latencies.append(time)
            else:
                cur_latencies.append(time)
    return np.array(cur_latencies)


