import numpy as np

def range_querying_fft(rf_data, bin_indices):
    """
    Perform range beamforming on RF data. We only query the specified range bins (can be fractional).

    Args:
        rf_data (numpy.ndarray): Input RF data. The last dimension should be samples.
        bin_indices (numpy.ndarray): Indices of range bins where beam should be formed.

    Returns:
        numpy.ndarray: Frequency domain data with the last dimension as specified range bins.
    """
    n_samples = rf_data.shape[-1]

    discrete_timestamps = np.arange(n_samples)
    base_matrix = np.exp(-1.0j * 2 * np.pi * np.outer(discrete_timestamps, np.array(bin_indices)) / n_samples)

    return rf_data @ base_matrix
