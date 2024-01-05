#!/usr/bin/env python3

from human_hip.latencies import getLatencies










def plotPairAnalysis(start_i, end_i, sd):
    
    # Layout Plot
    import warnings
    warnings.filterwarnings("ignore")
    plotPairFootprint( start_i, end_i, sd )
    
    # Summary Stats
    latencies_raw = getLatencies( start_i, end_i, sd  )
    latencies_clean = latencies_raw[ np.where( np.abs(latencies_raw) < 15 )[0] ]
    print( "Number of Latencies", len(latencies_clean) )
    print( "Mean Latency", round(np.mean(latencies_clean), 3) )
    print( "Median Latency", round(np.median(latencies_clean), 3) )
    #print( "Latency Probability", {round(100*(len(lates_filtered)/len(lates_raw)))})
    print("STTC", sd.spike_time_tiling( start_i, end_i) )
    print("Diptest P-val", round( diptest.diptest(latencies_clean)[1] , 3) )
    print( "Latency Probability", round(len(latencies_clean)/len(latencies_raw), 3) )
    print(".")
    print( "Sender Neuron", start_i)
    print( "Sender   Firing Rate", round(sd.rates(unit='Hz')[start_i] , 3) )
    print(".")
    print("Receiver Neuron", end_i)
    print( "Receiver Firing Rate", round(sd.rates(unit='Hz')[end_i], 3) )
    
    # Plot other graphs
    figs, plots = plt.subplots(nrows=3,ncols=2,figsize=(14,10))

    mean_latency = np.mean(latencies_clean)     # Get mean and SD
    std_latency = np.std(latencies_clean)
    cutoff = 2 * std_latency           # remove outliers
    lates_filtered = [latency for latency in latencies_clean if abs(latency - mean_latency) <= cutoff]
    plots[0,0].hist(lates_filtered, bins=12, alpha=0.7, label='Latency')
    plots[0,0].axvline(mean_latency, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plots[0,0].axvline(mean_latency - std_latency, color='green', linestyle='dashed', linewidth=2, label='Std -')
    plots[0,0].axvline(mean_latency + std_latency, color='green', linestyle='dashed', linewidth=2, label='Std +')
    plots[0,0].axvline(0, color='black', linestyle='dashed', linewidth=0.5, label='Std +')
    plots[0,0].set_xlim(-1*(abs(mean_latency)+cutoff), abs(mean_latency) + cutoff) 
    plots[0,0].set_xlabel("Latency (ms)")
    plots[0,0].set_ylabel("Count")
    plots[0,0].set_title(f"Fancy Latency Histogram")
    plots[0,0].legend()

    plots[0,1].hist(latencies_clean, bins=12)
    plots[0,1].set_title("Latency Histogram")
    plots[0,1].set_xlabel("Latency (ms)")
    plots[0,1].set_ylabel("Count")

    plots[1,0].hist(sd.interspike_intervals()[start_i], bins=50);
    plots[1,0].set_title("Sender ISI")
    plots[1,0].set_xlabel("Time bin(ms)")
    plots[1,0].set_ylabel("ISI count")

    plots[1,1].hist(sd.interspike_intervals()[end_i], bins=50);
    plots[1,1].set_title("Receiver ISI")
    plots[1,1].set_xlabel("Time bin(ms)")
    plots[1,1].set_ylabel("ISI count")
    
    
    plots[2,0].plot( sd.neuron_data[0][start_i]["template"] )
    plots[2,0].set_title("Sender Spike Waveform")
    plots[2,0].set_xlabel("")
    plots[2,0].set_ylabel("")
    
    plots[2,1].plot( sd.neuron_data[0][end_i]["template"] )
    plots[2,1].set_title("Receiver Spike Waveform")
    plots[2,1].set_xlabel("")
    plots[2,1].set_ylabel("")