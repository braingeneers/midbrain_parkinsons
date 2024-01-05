#!/usr/bin/env python3

import numpy as np                                                    # Packages for data analysis
import matplotlib.pyplot as plt
from human_hip.spike_data.analysis import correlation_matrix, eigenvalues_eigenvectors #firing_rates,

    
def reconstruct(W, U, rank):
    Wd = np.diag(W[:rank])
    Ur = U[:, :rank]
    return Ur @ Wd @ Ur.T


def reconstruction_errors(A, W, U):
    norm = np.linalg.norm(A)
    return np.array([
        np.linalg.norm( reconstruct(W, U, rank) - A) / norm
        for rank in range(len(A))])


def plot_eigenvectors(sd, n_eigenvectors=5 ):
    corr_matrix = correlation_matrix(sd)
    sttc_matrix = sd.spike_time_tilings()
    corr_eigenvalues, corr_eigenvectors = eigenvalues_eigenvectors(corr_matrix)
    sttc_eigenvalues, sttc_eigenvectors = eigenvalues_eigenvectors(sttc_matrix)
    
    fig, axs = plt.subplots(5, 2, figsize=(12,8)) #,  )

    for i in range(n_eigenvectors):
        #if i: plt.xticks([])
        axs[i,0].stem(sttc_eigenvectors[:,i])
        axs[i,1].stem(corr_eigenvectors[:,i])
        axs[i,0].set(ylabel= f"{i+1}")
        
        ylim_min = np.min([sttc_eigenvectors[:,i], corr_eigenvectors[:,i]]) -.1
        ylim_max = np.max([sttc_eigenvectors[:,i], corr_eigenvectors[:,i]]) +.1
        axs[i,0].set_ylim(ylim_min, ylim_max)
        axs[i,1].set_ylim(ylim_min, ylim_max)

    for ax in fig.get_axes():
        ax.label_outer()
    axs[0, 0].set_title("STTC")
    axs[0, 1].set_title("Correlation")



def plot_eigenvector_matrix( sd, plot_color="magma"):
    corr_matrix = correlation_matrix(sd)
    sttc_matrix = sd.spike_time_tilings()
    corr_eigenvalues, corr_eigenvectors = eigenvalues_eigenvectors(corr_matrix)
    sttc_eigenvalues, sttc_eigenvectors = eigenvalues_eigenvectors(sttc_matrix)
    fig, plots = plt.subplot_mosaic( """AB"""  , figsize=(14,7) )   # Set up layout for 2 figures,
    
    # subplot of STTC 
    pltA = plots["A"].imshow( sttc_eigenvectors.T, interpolation='none', cmap=plot_color)       # Show the STTC matrix
    plots["A"].set_title("STTC Eigenvectors")         # Set the title, x and y labels
    plots["A"].set_ylabel("Eigenvector")
    plots["A"].set_xlabel("Neuron")
    fig.colorbar(pltA, ax=plots["A"], shrink=0.5) # Add a colorbar to the plot
    
    # subplot of Correlation
    pltB = plots["B"].imshow( corr_eigenvectors.T, interpolation='none', cmap=plot_color)      # Show the correlation matrix
    plots["B"].set_title("Correlation Eigenvectors") # Set the title, x and y labels
    plots["B"].set_ylabel("Eigenvector")
    plots["B"].set_xlabel("Neuron")
    fig.colorbar(pltB, ax=plots["B"], shrink=0.5) # Add a colorbar to the plot



def plot_pca( sd, sttc=True ):
    corr_matrix = correlation_matrix(sd)
    sttc_matrix = sd.spike_time_tilings()
    corr_eigenvalues, corr_eigenvectors = eigenvalues_eigenvectors(corr_matrix)
    sttc_eigenvalues, sttc_eigenvectors = eigenvalues_eigenvectors(sttc_matrix)

    plt.figure(figsize=(7,5))
    plt.scatter( corr_eigenvectors[0], corr_eigenvectors[1], alpha=.5) # first two eigenvectors of correlation 
    if sttc:
        plt.scatter( sttc_eigenvectors[0], sttc_eigenvectors[1], alpha=.5) # first two eigenvectors of STTC
        plt.legend(["Correlation Eigenvectors", "STTC Eigenvectors"])
    plt.title("Principle Components  of Correlation Methods")
    plt.xlabel("PC1")
    plt.ylabel("PC2")



def plot_eigen_reconstrution( sd ):
    corr_matrix = correlation_matrix(sd)
    sttc_matrix = sd.spike_time_tilings()
    corr_eigenvalues, corr_eigenvectors = eigenvalues_eigenvectors(corr_matrix)
    sttc_eigenvalues, sttc_eigenvectors = eigenvalues_eigenvectors(sttc_matrix)
    
    fig, plot = plt.subplot_mosaic("AB", figsize=(15,5))     
    # Eigenvalue spectrum ---------------------------------------------------------
    index = 1 + np.arange(len(corr_eigenvalues))
    plot["A"].semilogy(index, corr_eigenvalues, label='Correlation')
    plot["A"].plot(index, sttc_eigenvalues, label='STTC')
    plot["A"].set_title('Eigenvalue Spectrum')
    plot["A"].set_xlabel('Index')
    plot["A"].set_ylabel('Eigenvalue')
    plot["A"].axvline(len(corr_eigenvalues)+1, ls=':', c='k')
    plot["A"].legend(loc="upper right");
    
    # Reconstruction Error --------------------------------------------------------
    errs_corr = reconstruction_errors(corr_matrix, corr_eigenvalues, corr_eigenvectors )      # Correlation
    errs_sttc = reconstruction_errors(sttc_matrix, sttc_eigenvalues, sttc_eigenvectors )     # STTC   
    
    plot["B"].plot(errs_corr, label='Correlation Matrix')          # Correlation
    plot["B"].plot(errs_sttc, label='STTC Matrix')                 # STTC
    
    plot["B"].set_title("Reconstruction Error of Correlation Methods")
    plot["B"].set_xlabel('Components Used in Reconstruction')
    plot["B"].set_ylabel('Relative Reconstruction Error')
    plot["B"].legend(loc="upper right");
    fig.show()



def plot_eigen_vector_layout(sd, vect, show_sttc=True, sttc_threshold=.1, plot_color="magma"):
    neuron_x = []
    neuron_y = []
    for neuron in sd.neuron_data[0].values(): # Plots neurons on a 2-d space, representing their positions on the array
        neuron_x.append(neuron['position'][0])
        neuron_y.append(neuron['position'][1])

    plt.figure(figsize=(10,7)) 
    ax = plt.axes()
    ax.set_facecolor("grey")
    plt.scatter(neuron_x,neuron_y,  c=vect, cmap=plot_color) # s=firing_rates(sd)*20, # color each plotted neuron according to the values of the eigenvector
   
    if show_sttc:
        sttc = sd.spike_time_tilings()
        for i in range(sttc.shape[0]): # plot connectivity lines between neurons
            for j in range(sttc.shape[1]):
                # Only need to do upper triangle since sttc' = sttc
                if i<=j: continue
                if sttc[i,j] < sttc_threshold : continue
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


def plot_eigendecomposition_vector(sd, vector_index=0, use_sttc=True, show_sttc=False, sttc_threshold=0.1,  plot_color="magma"):
    sd_matrix = sd.spike_time_tilings() if use_sttc else correlation_matrix(sd)
    eigenvalues, eigenvectors = eigenvalues_eigenvectors(sd_matrix)
    plot_eigen_vector_layout(sd, eigenvectors[:, vector_index], show_sttc, sttc_threshold,  plot_color )