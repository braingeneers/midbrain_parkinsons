#!/usr/bin/env python3

import numpy as np
import random as rand
from braingeneers.analysis.analysis import SpikeData, randomize_raster


def random_rotation(sd, seedIn=1): #seed=np.random.randint(0, 10000)
    # Randomizes a dataset while perseving an underlying structure by stitching sections of the train together at random points
    RotatedTrain = []
    TrainDat = sd.train
    
    np.random.seed(seedIn) # set random seed to seedIn
    seeds = [np.random.randint(0, 1000000000) for _ in range(len(TrainDat))] # creat array of randomized seeds from seedIn
    i = 0
    
    for neuron in TrainDat: # Loop through indiv neurons in train
        neuronTrain = []
        rand.seed(seeds[i]) # set random seed to one of seedIn's generated vals
        i += 1
        alpha = rand.randrange(1, len(neuron)) # Select random index
        
        index = alpha
        prevtime = 0 # global time
        while index < len(neuron): # Loop through neurons after alpha
            firingInst = neuron[index] - neuron[index - 1] # calculate firing time difference
            firingInst += prevtime # add difference to global time
            neuronTrain.append(firingInst)
            prevtime = firingInst # update global time
            index += 1
        index = 0
        splitEndVal = neuronTrain[-1] # get last time value of the split
        while index < alpha: # Loop through before alpha
            neuronTrain.append(neuron[index] + splitEndVal) # add times of pre-alpha neurons to the last time value and append 
            index += 1
        
        RotatedTrain.append(np.array(neuronTrain))
    return SpikeData( RotatedTrain, length=sd.length, N=sd.N, 
                         metadata=sd.metadata, neuron_data=sd.neuron_data,
                         neuron_attributes=sd.neuron_attributes) # convert to spikedata object and return



def random_shuffle(sd, seedIn=1): #seed=np.random.randint(0, 10000)
    # Randomizes a dataset while perseving an underlying structure by swaping firing times between neurons
    Rastered = sd.raster(bin_size=1) # get spikedata's raster
    RasterT = np.transpose(Rastered) # transpose raster to look at individual time slices
    shuffleT = []
    dt=1.0
  
    np.random.seed(seedIn) # set random seed to seedIn
    seeds = [np.random.randint(0, 1000000000) for _ in range(len(RasterT))] # creat array of randomized seeds from seedIn
    i = 0
    
    for timeslice in RasterT:
        randslice = np.copy(timeslice) 
        np.random.seed(seeds[i]) # set local seed to a seedIn generated value for randomized shuffle
        i += 1
        np.random.shuffle(randslice) # shuffle timeslice
        shuffleT.append(randslice) 
        
    ShuffleRaster = np.transpose(shuffleT) # re-transpose raster to get origional orientation 
    idces, times = np.nonzero(ShuffleRaster)
    
    return SpikeData( idces, times*dt, length=sd.length, N=sd.N, 
                         metadata=sd.metadata, neuron_data=sd.neuron_data,
                         neuron_attributes=sd.neuron_attributes)



def random_harris(sd, seed=1): #seed=None
        '''
        Kenneth Harris Method, Create a new SpikeData object which preserves the population
        rate and mean firing rate of each neuron in an existing
        SpikeData by randomly reallocating all spike times to different
        neurons at a resolution given by dt.
        '''
        # Collect the spikes of the original Spikedata and define a new
        # "randomized spike matrix" to store them in.
        sm = sd.sparse_raster(1.0)     # spike raster with 1ms bins 
        idces, times = np.nonzero(randomize_raster(sm, seed))
        return SpikeData( idces, times*dt, length=sd.length, N=sd.N,
                         metadata=sd.metadata, neuron_data=sd.neuron_data,
                         neuron_attributes=sd.neuron_attributes)