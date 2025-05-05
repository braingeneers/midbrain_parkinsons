#!/usr/bin/env python3
"""
Rewritten script to process each dataset sequentially.
For each dataset, we:
  - Download and load the SpikeData object.
  - Compute the original STTC matrix.
  - For each iteration:
      * Randomize the data (without storing the raster).
      * Create a new SpikeData object.
      * Compute its STTC matrix and write it into a memmapped array.
      * Save the SpikeData object in a dictionary.
  - Once done, convert the memmapped STTC matrices to a numpy array.
  - Compute a filtered STTC matrix (using the 95th percentile).
  - Assemble a dictionary with the original and randomized results.
  - Save and upload the dataset result, then clear memory.
"""

import os
import sys
import argparse
import shutil
import numpy as np
import braingeneers.utils.s3wrangler as wr
import gc
import random as rand

# Import your SpikeData loader and helper functions.
from load_data.load_npz import load_spikedata

def compute_true_matrices(sd, bin_size=1):
    original_raster = sd.raster(bin_size=bin_size)
    true_sttc = sd.spike_time_tilings()
    return original_raster, true_sttc

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
    return sd.__class__(RotatedTrain, length=sd.length, N=sd.N, 
                      metadata=sd.metadata, neuron_data=sd.neuron_data,
                      neuron_attributes=sd.neuron_attributes) # convert to spikedata object and return


def filter_true_matrix(true_matrix, sttc_array):
    """
    Compute the 95th percentile (across iterations) for each neuron pair,
    then retain the true STTC value only if it exceeds that percentile.
    """
    percentile95 = np.percentile(sttc_array, 95, axis=0)
    filtered = np.where(true_matrix >= percentile95, true_matrix, 0)
    return filtered

def process_dataset(infile, local_input_dir, local_output_dir, iterations):
    dataset_name = os.path.splitext(os.path.basename(infile))[0]
    print(f"Processing dataset: {dataset_name}")
    local_file = os.path.join(local_input_dir, f"{dataset_name}.npz")

    # Download or copy the input file locally.
    if infile.startswith("s3://"):
        print(f"Downloading {infile} to {local_file}")
        try:
            wr.download(infile, local_file)
        except Exception as e:
            print(f"Error downloading {infile}: {e}")
            sys.exit(1)
    else:
        try:
            shutil.copy(infile, local_file)
        except Exception as e:
            print(f"Error copying {infile}: {e}")
            sys.exit(1)

    # Load the SpikeData object.
    sd = load_spikedata(local_file)
    
    # Compute original raster (needed for randomization) and true STTC matrix.
    orig_raster, true_sttc = compute_true_matrices(sd, bin_size=1)

    # Prepare to store randomized STTC matrices via np.memmap.
    sttc_shape = true_sttc.shape  # (N, N)
    sttc_dtype = true_sttc.dtype
    memmap_file = os.path.join(local_output_dir, f"{dataset_name}_sttc_rr.dat")
    sttc_memmap = np.memmap(memmap_file, dtype=sttc_dtype, mode="w+", 
                            shape=(iterations, sttc_shape[0], sttc_shape[1]))

    # Dictionary to store SpikeData objects for each iteration.
    random_sd = {}

    # Process each iteration.
    for i in range(1, iterations + 1):
        seed = 42 + i
        # Randomize the original raster.
        new_sd = random_rotation(sd, seedIn=seed)
        # Store the SpikeData object.
        random_sd[i] = new_sd
        # Compute the STTC matrix for this randomized dataset.
        sttc_memmap[i - 1, :, :] = new_sd.spike_time_tilings()
        print(f"Iteration {i} complete for method raster")

    # Flush the memmap to disk.
    sttc_memmap.flush()
    # Convert the memmapped array to a regular numpy array.
    sttc_rand = np.array(sttc_memmap)

    # Compute the filtered STTC matrix using the 95th percentile across iterations.
    filtered_rr = filter_true_matrix(true_sttc, sttc_rand)

    # Assemble the results dictionary.
    dataset_dict = {
        "original": {"sd": sd, "sttc": true_sttc},
        "random": {"sd": random_sd, "sttc": sttc_rand, "filtered": filtered_rr}
    }

    # Save the dictionary as a compressed npz file.
    out_file = os.path.join(local_output_dir, f"{dataset_name}_results.npz")
    np.savez_compressed(out_file, **dataset_dict)
    print(f"Dataset {dataset_name} processing complete. Saved results to {out_file}")

    # Optionally, upload the result file to S3.
    s3_output_file = os.path.join(args.output_s3, os.path.basename(out_file))
    print(f"Uploading {out_file} to {s3_output_file}")
    try:
        wr.upload(out_file, s3_output_file)
    except Exception as e:
        print(f"Error uploading {out_file}: {e}")
        sys.exit(1)

    # Clean up large objects and remove local files if desired.
    del sd, orig_raster, true_sttc, sttc_memmap, sttc_rand, random_sd, dataset_dict
    gc.collect()

def main():
    parser = argparse.ArgumentParser(
        description="Randomize SpikeData objects, compute and filter STTC matrices, and upload results to S3."
    )
    parser.add_argument("input_files", nargs="+", help="Input .npz file paths (S3 or local) for SpikeData objects.")
    parser.add_argument("output_s3", help="Output S3 path for the results npz file (e.g. s3://bucket/output_folder)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of randomizations per method (default 100)")
    global args
    args = parser.parse_args()
    
    input_paths = args.input_files
    iterations = args.iterations

    # Setup local directories.
    local_input_dir = "/tmp/local_inputs"
    local_output_dir = "/tmp/output_results"
    os.makedirs(local_input_dir, exist_ok=True)
    os.makedirs(local_output_dir, exist_ok=True)
    
    # Process each dataset individually.
    for infile in input_paths:
        process_dataset(infile, local_input_dir, local_output_dir, iterations)
    
    print("Analysis complete. All datasets processed and uploaded.")

if __name__ == '__main__':
    main()
