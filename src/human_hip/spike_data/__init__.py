from .read_phy_files import read_phy_files
from .analysis import correlation_matrix, firing_rates, ISI, eigenvalues_eigenvectors, instant_firing_rate
from .plot_raster import plot_raster_fancy, plot_raster
from .text_summary import text_summary, text_summary_UUID
from .plot_summary import plot_summary
from .plot_matrices_connectivity import correlation_matrix, plot_sttc_matrix, plot_correlation_matrix, plot_matrices_connectivity
from .plot_functional_connectivity_map import plot_functional_connectivity_map
from .plot_eigendecomposition import plot_eigenvectors, plot_eigenvector_matrix, plot_pca, plot_eigen_reconstrution, \
                                        plot_vector_layout, plot_eigendecomposition_vector
from .plot_footprint import plot_footprint


__all__ = [ "read_phy_files", "correlation_matrix", "firing_rates", "ISI",  "eigenvalues_eigenvectors", "instant_firing_rate",
            "plot_raster_fancy", "plot_raster", "text_summary", "text_summary_UUID", "instant_firing_rate", "plot_summary",
            "plot_sttc_matrix", "plot_correlation_matrix", "plot_matrices_connectivity", "plot_functional_connectivity_map",
            "plot_eigenvectors", "plot_eigenvector_matrix", "plot_pca", "plot_eigen_reconstrution", "plot_vector_layout",
            "plot_eigendecomposition_vector", "plot_footprint"] 

#from ..example import example_func # example of looking one directory up for a command

