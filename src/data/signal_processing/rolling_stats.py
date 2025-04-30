import numpy as np
import numba as nb

from typing import Literal

from src.core import validate_literal_args


@nb.njit()
def _rolling_quantile(padded_data, original_data_size, window, quantile):
    """
    Numba accelerated function to calculate rolling quantile.
    """
    quantiles = np.zeros((original_data_size,))
    for i in range(original_data_size):
        sub_array = padded_data[i:i + window]
        sub_array = sub_array[~np.isnan(sub_array)]
        if sub_array.shape[0] == 0:
            quantiles[i] = quantiles[i - 1]
        else:
            quantiles[i] = np.quantile(sub_array, quantile)
    return quantiles


@nb.njit()
def _rolling_std(padded_data, original_data_size, window, quantile):
    """
    Numba accelerated function to calculate rolling standard deviation.
    """
    std = np.zeros((original_data_size,))
    for i in range(original_data_size):
        sub_array = padded_data[i:i + window]
        sub_array = sub_array[~np.isnan(sub_array)]
        if sub_array.shape[0] == 0:
            std[i] = std[i - 1]
        else:
            # Calculate standard deviation only for values below the quantile
            bar = np.quantile(sub_array, quantile)
            std[i] = np.std(sub_array[sub_array < bar])
    return std


@validate_literal_args
def rolling_stats(data: np.ndarray, stat_type: Literal["std", "quantile"], window=5, quantile=0.5):
    """
    Calculate rolling statistics: a sliding window of hop size 1.

    Args:
        data: Input data.
        stat_type: Type of statistic to calculate. Either "std" or "quantile".
        window: Window size for rolling statistics.
        quantile: The quantile value. When stat_type is "std", this value is used as a threshold.

    Returns:
        Rolling statistics.
    """
    # Pad data to handle borders
    if window % 2 == 0:
        window += 1
    pad_width = window // 2
    padded_data = np.pad(data, (pad_width, pad_width), mode='reflect')
    if stat_type == "std":
        return _rolling_std(padded_data, len(data), window, quantile)
    elif stat_type == "quantile":
        return _rolling_quantile(padded_data, len(data), window, quantile)
