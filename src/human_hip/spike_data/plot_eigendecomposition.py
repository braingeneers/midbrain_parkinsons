#!/usr/bin/env python3



"""
####################
OlD imports
####################
"""

import pytz
import numpy as np                                                    # Packages for data analysis
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
import braingeneers                                                   # Braingeneers code
import ipywidgets as ipw
from ipywidgets import interact, interactive, fixed, interact_manual  # package for interactive widgets from IPython.display import HTML, display, Javascript, clear_output,
from IPython.display import HTML, display, Javascript, clear_output

import braingeneers.data.datasets_electrophysiology as ephys
from braingeneers.analysis.analysis import SpikeData, read_phy_files, load_spike_data, burst_detection, randomize_raster
from human_hip.spike_data.analysis import correlation_matrix


"""
####################
Paramters
####################
"""
plot_color = 'magma'   # Other colors: 'viridis' 'plasma' 'inferno' 'magma' 




"""
####################
Old Helper Functions
####################
"""


def plot_raster(sd, title="Spike Raster", l1=-10, l2=False, xsize=10, ysize=6, analize=False):
    if l2==False:
        l2 = sd.length / 1000 + 10
    
    idces, times = sd.idces_times()
    
    if analize == True:
        # Get population rate for everything
        pop_rate = sd.binned(bin_size=1)  # in ms
        # Lets smooth this to make it neater
        sigma = 5
        pop_rate_smooth = gaussian_filter1d(pop_rate.astype(float), sigma=sigma)
        t = np.linspace(0, sd.length, pop_rate.shape[0]) / 1000

        # Determine the stop_time if it's not provided
        if l2 is None:
            l2 = t[-1]

        # Filter times and idces within the specified start and stop times
        mask = (times >= l1 * 1000) & (times <= l2 * 1000)
        times = times[mask]
        idces = idces[mask]

    fig, ax = plt.subplots(figsize=(xsize, ysize))
    fig.suptitle(title)
    ax.scatter(times/1000,idces,marker='|',s=1)
    
    if analize == True:
        ax2 = ax.twinx()
        ax2.plot(t, pop_rate_smooth, c='r')
        ax2.set_ylabel('Firing Rate')
        
    ax.set_xlabel("Time(s)")
    ax.set_ylabel('Unit #')
    plt.xlim(l1, l2)
    plt.show()
    
def eigenvalues_eigenvectors(A):
    W, U = np.linalg.eigh(A)
    # The rank of A can be no greater than the smaller of its
    # dimensions, so cut off the returned values there.
    rank = min(*A.shape)
    U = U[:,-rank:]
    sgn = (-1)**(U[0,:] < 0)
    # Also reverse the order of the eigenvalues because eigh()
    # returns them in ascending order but descending makes more sense.
    return W[-rank:][::-1], (U*sgn[np.newaxis,:])[:, ::-1]

def toeplitz(M):
    K = M.shape[0]
    topz = np.zeros_like(M)
    for i in range(1 - K, K):
        meantopz = np.mean(np.diagonal(M, offset=i))
        topz += np.eye(K, K, k=i) * meantopz
        topz += np.eye(K, K, k=-i) * meantopz
    return topz
        
def get_sttc(sd):
    sttc = sd.spike_time_tilings()
    return sttc
    
def reconstruct(W, U, rank):
    Wd = np.diag(W[:rank])
    Ur = U[:, :rank]
    return Ur @ Wd @ Ur.T
    
def reconstruction_errors(A, ranks):
    norm = np.linalg.norm(A)
    W, U = eigenvalues_eigenvectors(A)
    return np.array([
        np.linalg.norm( reconstruct(W, U, rank) - A) / norm
        for rank in ranks])

def calculate_mean_firing_rates(spike_data):
    mean_firing_rates = []
    for neuron_spikes in spike_data.train:
        num_spikes = len(neuron_spikes)
        time_duration = spike_data.length / 1000  # Assuming spike times are in milliseconds
        firing_rate = num_spikes / time_duration
        mean_firing_rates.append(firing_rate)
    return np.array(mean_firing_rates)




"""
####################
Old Subplots
####################
"""


def corrMethPlots(Corr, STTC):
    fig, plot = plt.subplot_mosaic("AB", figsize=(14,7)) 
    
    # Plot Correlation Matrix
    pltA = plot["A"].imshow(Corr, cmap=plot_color) # plot STTC matrix
    plot["A"].set_title("Correlation")
    plot["A"].set_xlabel("Neuron index")
    plot["A"].set_ylabel("Neuron index" )
    fig.colorbar(pltA, ax=plot["A"], shrink=0.7)

    # Plot STTC matrix
    pltB = plot["B"].imshow(STTC, cmap=plot_color) # plot STTC matrix
    plot["B"].set_title("STTC")
    plot["B"].set_xlabel("Neuron index")
    plot["B"].set_ylabel("Neuron index" )
    fig.colorbar(pltB, ax=plot["B"], shrink=0.7)




def EigenvectorAnalysis(corr, sttc):
    """figLayout = 
                AAABBB
                CCCDDD
                CCCDDD
                CCCDDD
                """
    
    figLayout = """
                AABB
                CCDD
                CCDD
                """
    
    fig, plot = plt.subplot_mosaic(figLayout, figsize=(16,8))
    fig.tight_layout(pad=5.0)
    
    # Plot Correlation basis
    plot["A"].stem(corr[:,0])
    plot["A"].set_xlabel('Observation Dimension')
    plot["A"].set_title('First Eigenvector of Correlation')    
    
    # Plot STTC basis
    plot["B"].stem(sttc[:,0])
    plot["B"].set_xlabel('Observation Dimension')
    plot["B"].set_title('First Eigenvector of STTC')    
    
    # Plot Correlation Matrix
    pltC = plot["C"].imshow(corr[:,:2*len(corr)].T, interpolation='none', cmap=plot_color)
    #plot["A"].gca().set_aspect('auto')
    plot["C"].set_ylabel('Eigenvector Number')
    plot["C"].set_xlabel('Observation Dimension')
    plot["C"].set_title('Eigenvectors of Correlation')
    fig.colorbar(pltC, ax=plot["C"], shrink=0.7)
    
    # Plot STTC matrix
    pltD = plot["D"].imshow(sttc[:,:2*len(sttc)].T, interpolation='none', cmap=plot_color)
    #plot["B"].gca().set_aspect('auto')
    plot["D"].set_ylabel('Eigenvector Number')
    plot["D"].set_xlabel('Observation Dimension')
    plot["D"].set_title('Eigenvectors of STTC')
    fig.colorbar(pltD, ax=plot["D"], shrink=0.7)




def plot_evectmatrix(corr, sttc):
    fig, plot = plt.subplot_mosaic("AB", figsize=(14,7))
    
    # Plot Correlation Matrix
    pltA = plot["A"].imshow(corr[:,:2*len(corr)].T, interpolation='none', cmap=plot_color)
    #plot["A"].gca().set_aspect('auto')
    plot["A"].set_ylabel('Eigenvector Number')
    plot["A"].set_xlabel('Observation Dimension')
    plot["A"].set_title('Eigenvectors of Correlation')
    fig.colorbar(pltA, ax=plot["A"], shrink=0.7)
    
    # Plot STTC matrix
    pltA = plot["B"].imshow(sttc[:,:2*len(sttc)].T, interpolation='none', cmap=plot_color)
    #plot["B"].gca().set_aspect('auto')
    plot["B"].set_ylabel('Eigenvector Number')
    plot["B"].set_xlabel('Observation Dimension')
    plot["B"].set_title('Eigenvectors of STTC')
    fig.colorbar(pltA, ax=plot["B"], shrink=0.7)



def plot_basis(method):
    
    if method == "Correlation":
        Corr = correlation(sd)
        Wcorr, Ucorr = eigenvalues_eigenvectors(Corr)
        A = Ucorr
    else:
        STTC = get_sttc(sd)
        Wsttc, Usttc = eigenvalues_eigenvectors(STTC)
        A = Usttc
        
    plt.figure(figsize=(6,8))
    for i in range(5):
        if i: plt.xticks([])
        plt.subplot(5, 1, i+1)
        plt.stem(A[:,i])
    plt.xlabel('Observation Dimension')
    plt.suptitle('First 5 Eigenvectors', y=0.92)   



def PCAplots(corr, sttc):
    plt.figure(figsize=(7,5))
    plt.scatter(corr[0], corr[1]) # first two eigenvectors of correlation 
    plt.scatter(sttc[0], sttc[1]) # first two eigenvectors of STTC
    plt.legend(["Correlation Eigenvectors", "STTC Eigenvectors"])
    plt.title("Principle Components Analysis of Correlation Methods")
    plt.xlabel("PC1")
    plt.ylabel("PC2")


def ReconstructPlots(Wcorr, Wsttc, Corr, STTC):
    fig, plot = plt.subplot_mosaic("AB", figsize=(15,5)) 
    
    # Eigenvalue spectrum ---------------------------------------------------------
    index = 1 + np.arange(len(Wcorr))
    plot["A"].semilogy(index, Wcorr, label='Correlation')
    plot["A"].plot(index, Wsttc, label='STTC')
    plot["A"].set_title('Eigenvalue Spectrum')
    plot["A"].set_xlabel('Index')
    plot["A"].set_ylabel('Eigenvalue')
    plot["A"].axvline(len(Wcorr)+1, ls=':', c='k')
    plot["A"].legend(loc="upper right");
    
    # Reconstruction Error --------------------------------------------------------
    errs_corr = reconstruction_errors(Corr, range(len(Corr)))      # Correlation
    errs_sttc = reconstruction_errors(STTC , range(len(STTC)))     # STTC   
    
    plot["B"].plot(errs_corr, label='Correlation Matrix')          # Correlation
    plot["B"].plot(errs_sttc, label='STTC Matrix')                 # STTC
    
    plot["B"].set_title("Reconstruction Error of Correlation Methods")
    plot["B"].set_xlabel('Components Used in Reconstruction')
    plot["B"].set_ylabel('Relative Reconstruction Error')
    plot["B"].legend(loc="upper right");
    fig.show()



def evectLayout(sd, vect, threshold=0):
    #x = electrode_mapping.x.values
    #y = electrode_mapping.y.values

    sttc = sd.spike_time_tilings()
    firing_rates = calculate_mean_firing_rates(sd)
    
    neuron_x = []
    neuron_y = []
    #neuron_amp = []
    for neuron in sd.neuron_data[0].values(): # Plots neurons on a 2-d space, representing their positions on the array
    #     print("x,y:",neuron['position'])
        neuron_x.append(neuron['position'][0])

        neuron_y.append(neuron['position'][1])
        #neuron_amp.append(np.mean(neuron['amplitudes']))

    # 
    plt.figure(figsize=(7,5)) 
    ax = plt.axes()
    ax.set_facecolor("grey")
    #plt.scatter(neuron_x,neuron_y, s=neuron_amp, c=vect, cmap = plot_color) # color each plotted neuron according to the values of the eigenvector
    plt.scatter(neuron_x,neuron_y, s=firing_rates*20, c=vect, cmap = plot_color) # color each plotted neuron according to the values of the eigenvector
   
    
    for i in range(sttc.shape[0]): # plot connectivity lines between neurons
        for j in range(sttc.shape[1]):

            # Only need to do upper triangle since sttc' = sttc
            if i<=j: continue

            if sttc[i,j] < threshold : continue

            #Position of neuron i
            ix,iy = sd.neuron_data[0][i]['position']
            jx,jy = sd.neuron_data[0][j]['position']

            # Plot line between the points, linewidth is the sttc
            plt.plot([ix,jx],[iy,jy], linewidth=sttc[i,j],c='k')
            
    plt.xlabel('um')
    plt.ylabel('um')
    plt.title("Neuron layout")
    plt.colorbar()
    plt.show()



def EigenvectorLayout(Method):
    global ELayout_usemethod;
    
    if Method == "Correlation":
        Corr = correlation(sd)
        Wcorr, Ucorr = eigenvalues_eigenvectors(Corr)
        ELayout_usemethod = Ucorr
    else:
        STTC = get_sttc(sd)
        Wsttc, Usttc = eigenvalues_eigenvectors(STTC)
        ELayout_usemethod = Usttc
        
    interact_manual( vectorSelect, Vector=(0,len(ELayout_usemethod),1), thresholdIn=(0,1,0.01)) 


def vectorSelect(Vector, thresholdIn):
    evectLayout(sd, ELayout_usemethod[:,Vector], thresholdIn)


def vectorSelectV2(Vector):
    evectLayout(sd, ELayout_usemethod[:,Vector])
