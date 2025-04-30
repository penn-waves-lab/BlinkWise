from typing import Literal

import numpy as np
from numpy import ma

from src.core import validate_literal_args

def _calculate_stat(window, stat_func, axis):
    if stat_func == "std":
        return np.std(window, axis=axis)
    elif stat_func == "mean":
        return np.mean(window, axis=axis)
    elif stat_func == "max":
        return np.max(window, axis=axis)
    else:
        raise ValueError(f"Unsupported stat_func: {stat_func}")


@validate_literal_args
def window_stats(
        data,
        window_size,
        stat_func: Literal["std", "mean", "max"],
        axis: int = 0,
        hop: int = None,
        apply_ema=False,
        mask_valid_entries=None,
        lagged_stats: bool = False,
        return_full_size: bool = False,
):
    """
    calculate statistics over sliding windows of a time series data array.

    Args:
        data (numpy.ndarray): Time series data.
        window_size (int): Size of the window to calculate statistics (in frames).
        stat_func (Literal["std", "mean", "max"]): Statistic to calculate ("std", "mean", "max").
        axis (int, optional): Axis to calculate statistics along (default is 0).
        hop (int, optional): Hop size for sliding windows (default is None for non-overlapping windows).
        apply_ema (bool, optional): Apply Exponential Moving Average smoothing (default is False).
        mask_valid_entries (numpy.ndarray, optional): Boolean mask to specify valid frames for statistics calculation (default is None).
        lagged_stats (bool, optional): Use previous window statistics for current window (default is False).
        return_full_size (bool, optional): Repeat windowed statistics to match original data size (default is False).

    Returns:
        numpy.ndarray: Array of calculated statistics.
    """
    # determine hop size
    if hop is None:
        hop = window_size

    # apply mask
    if mask_valid_entries is not None:
        data = ma.masked_array(data, mask=(~mask_valid_entries))

    # #windows x window_size x remaining dimensions
    n_windows = (data.shape[axis] - window_size) // hop + 1
    windowed_stats = []

    for i in range(n_windows):
        start_idx = i * hop
        end_idx = start_idx + window_size

        if end_idx > data.shape[axis]:
            break

        window = np.take(data, indices=range(start_idx, end_idx), axis=axis)
        stat = _calculate_stat(window, stat_func, axis)
        windowed_stats.append(stat)

    windowed_stats = np.array(windowed_stats)

    # check if any window is fully masked
    if mask_valid_entries is not None:
        if np.any(np.isnan(windowed_stats)):
            raise ValueError("In some windows no valid data points appear. Try setting a larger window.")

    # apply ema if specified
    if apply_ema:
        alpha = 0.3
        ema_stats = [windowed_stats[0]]
        for stat in windowed_stats[1:]:
            ema_stat = alpha * stat + (1 - alpha) * ema_stats[-1]
            ema_stats.append(ema_stat)
        windowed_stats = np.array(ema_stats)

    # apply lagged stats if specified
    if lagged_stats:
        lagged_stats = [windowed_stats[0]]
        for stat in windowed_stats[0:-1]:
            lagged_stats.append(stat)
        windowed_stats = np.array(lagged_stats)

    # return full size if specified
    if return_full_size:
        full_size_stats = np.empty(data.shape)
        # the first window will be filled with its own statistics
        full_size_stats[:window_size] = windowed_stats[0]
        # print(f"{0}->{window_size}: {windowed_stats[0]}")

        # every hop frames after the first window will use the stat in its corresponding window.
        # when lagged, since the windowed_stats have been lagged, it will fit the following codes
        for i in range(1, n_windows):
            start_idx = window_size + (i - 1) * hop
            end_idx = start_idx + hop
            # print(f"{start_idx}->{end_idx}: {windowed_stats[i]}")
            full_size_stats[start_idx:end_idx] = windowed_stats[i]

        # handle remaining elements
        remaining_start_idx = window_size + (n_windows - 1) * hop
        if remaining_start_idx < data.shape[axis]:
            full_size_stats[remaining_start_idx:] = windowed_stats[-1]

        return full_size_stats

    return windowed_stats
