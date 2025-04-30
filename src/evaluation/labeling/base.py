from abc import abstractmethod
from pathlib import Path
from typing import Union, Optional

import numpy as np
from scipy import signal, ndimage

from src.core import constants, blink_defs
from src.data.signal_processing import rolling_stats
from ..utils import BaseProcessor


class AutoLabeler(BaseProcessor):
    def __init__(
            self,
            exp_config_path: Union[str, Path],
            peak_quantile: float = 0.85,
            ptp_quantile: float = 0.95,
            closing_wlen: float = 0.3,
            reopening_wlen: float = 0.5,
            quantile_window: float = 30,
            fps: int = constants.ROUNDED_FPS,
            min_peak_distance: float = 0.2,
            output_folder: Optional[Union[str, Path]] = None,
            verbose: bool = True,
    ):
        """
        Base class for automatic blink labeling.

        Blinks are detected using the velocity of the blink ratio, and then subclasses determines phases with different methods.

        Args:
            exp_config_path: Path to the experiment configuration file.
            peak_quantile: Quantile value to be used as the minimum height for peak detection in blink velocity.
            ptp_quantile: Quantile value to be used to as the minimum disparity between closing and reopening peaks.
            closing_wlen: Window length for peaks that happen during eye closing.
            reopening_wlen: Window length for peaks that happen during eye reopening.
            quantile_window: Window length for quantile calculation.
            fps: Frames per second.
            min_peak_distance: Minimum distance between peaks.
            output_folder: Path to the output folder.
            verbose: Whether to print logs.
        """
        super().__init__(name="labeling", exp_config_path=exp_config_path, output_folder=output_folder, verbose=verbose)

        self.peak_quantile = peak_quantile
        self.ptp_quantile = ptp_quantile
        self.closing_wlen = closing_wlen
        self.reopening_wlen = reopening_wlen
        self.quantile_window = quantile_window
        self.fps = fps
        self.min_peak_distance = min_peak_distance

        self.verbose = verbose

    @property
    def quantile_window_frames(self):
        return int(self.quantile_window * self.fps)

    @property
    def closing_wlen_frames(self):
        return int(self.closing_wlen * self.fps)

    @property
    def reopening_wlen_frames(self):
        return int(self.reopening_wlen * self.fps)

    @property
    def min_peak_distance_frames(self):
        return int(self.min_peak_distance * self.fps)

    @staticmethod
    def get_clusters_closed_interval(indices: np.ndarray[int]) -> list[tuple[int, int]]:
        """
        Get clusters of indices, inclusive.

        Clusters are defined as continuous indices.

        Args:
            indices: Indices.

        Returns:
            list: List of tuples of start and end indices of clusters, inclusive.
        """
        starts = indices[np.where(np.diff(indices, prepend=-2) > 1)[0]]
        ends = indices[np.where(np.diff(indices, append=indices[-1] + 2) > 1)[0]]
        return list(zip(starts, ends))

    def find_and_match_peaks(
            self,
            blink_velocity: np.ndarray[float],
            non_blinking_mask: np.ndarray[bool]
    ) -> tuple[list[int], list[int]]:
        """
        Find, match, and filter closing and reopening peaks in the blink velocity. Matched pairs indicate blinks.

        We first find closing and reopening peaks in the blink velocity. Closing corresponds to negative peaks (openness decreases) and reopening corresponds to positive peaks (openness increases).

        We then match closing and reopening peaks by finding the closest pair of peaks. Closing should happen before reopening.

        We filter the pairs by the disparity between the closing and reopening peaks. The disparity should be larger than a threshold defined by ``ptp_quantile``.

        Args:
            blink_velocity: The first derivative of the blink ratio.
            non_blinking_mask: Mask for non-blinking frames.

        Returns:
            closing_v_peaks: Indices of the closing peaks.
            reopening_v_peaks: Indices of the reopening peaks.
        """
        blink_velocity_zero_masked = blink_velocity.copy()
        # variations in inter-blink intervals requires us to mask the non-blinking frames to get a consistent quantile value across subjects
        blink_velocity_zero_masked[non_blinking_mask] = np.nan

        peak_quantile_values = rolling_stats(
            blink_velocity_zero_masked,
            "quantile",
            self.quantile_window_frames,
            self.peak_quantile,
        )
        ptp_quantile_values = rolling_stats(
            blink_velocity_zero_masked,
            "quantile",
            self.quantile_window_frames,
            self.ptp_quantile,
        )

        # find peaks
        reopening_v_peaks, _ = signal.find_peaks(
            blink_velocity,
            height=peak_quantile_values,
            wlen=self.reopening_wlen_frames,
            distance=self.min_peak_distance_frames,
        )
        closing_v_peaks, _ = signal.find_peaks(
            -blink_velocity,
            height=peak_quantile_values,
            wlen=self.closing_wlen_frames,
            distance=self.min_peak_distance_frames,
        )

        # match and filter peaks
        peak_distance = closing_v_peaks.reshape((-1, 1)) - reopening_v_peaks.reshape((1, -1))
        peak_distance_abs = np.abs(peak_distance)
        peak_distance_negative = np.ma.masked_array(peak_distance_abs, mask=~(peak_distance < 0))

        min_row_mask = (peak_distance_negative == np.ma.min(peak_distance_negative, axis=1).data[:, np.newaxis]).data
        min_col_mask = (peak_distance_negative == np.ma.min(peak_distance_negative, axis=0).data[np.newaxis, :]).data

        velocity_disparity = np.abs(
            blink_velocity[closing_v_peaks].reshape((-1, 1))
            - blink_velocity[reopening_v_peaks].reshape((1, -1))
        )
        velocity_disparity_mask = velocity_disparity > 1.0 * np.tile(
            ptp_quantile_values[closing_v_peaks].reshape((-1, 1)),
            reopening_v_peaks.shape[0],
        )

        closing_v_peak_indices, reopening_v_peak_indices = np.where(
            min_row_mask & min_col_mask & velocity_disparity_mask
        )

        return [closing_v_peaks[i] for i in closing_v_peak_indices], [reopening_v_peaks[i] for i in
                                                                      reopening_v_peak_indices]

    def get_blinks_from_cluster(
            self,
            blink_ratio_inverse: np.ndarray[float],
            blink_velocity: np.ndarray[float],
            cluster_start: int,
            cluster_end: int,
            closing_v_peaks: list[int],
            reopening_v_peaks: list[int],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """
        Identify blinks from a cluster of positive frames.

        Multiple blinks can happen in one cluster. This function separates them.

        Returns:
            A tuple of lists of indices for the left most, right most, closing, and reopening points of the blinks.
        """
        closing_v_peaks_in_cluster = [k for k in closing_v_peaks if cluster_start < k < cluster_end]
        reopening_v_peaks_in_cluster = [k for k in reopening_v_peaks if cluster_start < k < cluster_end]

        if len(closing_v_peaks_in_cluster) != len(reopening_v_peaks_in_cluster):
            cluster_start = max(0, cluster_start - 100)
            cluster_end = min(blink_velocity.shape[0], cluster_end + 100)
            closing_v_peaks_in_cluster = [k for k in closing_v_peaks if cluster_start <= k <= cluster_end]
            reopening_v_peaks_in_cluster = [k for k in reopening_v_peaks if cluster_start <= k <= cluster_end]

        n_closing_v_peaks = len(closing_v_peaks_in_cluster)
        n_reopening_v_peaks = len(reopening_v_peaks_in_cluster)
        min_n_peaks = min(n_closing_v_peaks, n_reopening_v_peaks)

        if n_closing_v_peaks != n_reopening_v_peaks:
            self.logger.info("Closing (#{}) and reopening (#{}) peaks are not matched in cluster {}->{}!".format(
                len(closing_v_peaks_in_cluster), len(reopening_v_peaks_in_cluster), cluster_start, cluster_end
            ))
        if min_n_peaks == 0:
            return [], [], [], []

        closing_v_peaks_in_cluster = closing_v_peaks_in_cluster[-min_n_peaks:]
        reopening_v_peaks_in_cluster = reopening_v_peaks_in_cluster[-min_n_peaks:]

        # remove some reopening/closing peaks if one long blink is broken into two pieces
        to_remove = list(map(
            lambda paired_peaks: paired_peaks[1] - paired_peaks[0] < 100 and np.all(
                blink_ratio_inverse[paired_peaks[0]:paired_peaks[1]] > 0.4),
            zip(reopening_v_peaks_in_cluster[:-1], closing_v_peaks_in_cluster[1:])
        ))

        closing_v_peaks_in_cluster = [closing_v_peaks_in_cluster[0]] + [p for i, p in
                                                                        enumerate(closing_v_peaks_in_cluster[1:]) if
                                                                        not to_remove[i]]
        reopening_v_peaks_in_cluster = [p for i, p in enumerate(reopening_v_peaks_in_cluster[:-1]) if
                                        not to_remove[i]] + [reopening_v_peaks_in_cluster[-1]]

        # bound blinks
        left_most_indices = []
        right_most_indices = []
        closing_v_peaks_final = []
        reopening_v_peaks_final = []
        for pair_ind, (closing_v_peak, reopening_v_peak) in enumerate(
                zip(closing_v_peaks_in_cluster, reopening_v_peaks_in_cluster)):
            if pair_ind > 0:
                prev_reopening_peak = reopening_v_peaks_in_cluster[pair_ind - 1]
            else:
                prev_reopening_peak = cluster_start

            if pair_ind < len(closing_v_peaks_in_cluster) - 1:
                next_closing_peak = closing_v_peaks_in_cluster[pair_ind + 1]
            else:
                next_closing_peak = cluster_end

            # left most point is defined as the first point in the last cluster that has a negative velocity
            left_most_index = self.get_clusters_closed_interval(
                np.where(blink_velocity[prev_reopening_peak:closing_v_peak] < 0)[0]
            )[-1][0] + prev_reopening_peak

            # right most point is defined as the lst point in the first cluster that has a positive
            right_most_index = self.get_clusters_closed_interval(
                np.where(blink_velocity[reopening_v_peak:next_closing_peak] > 0)[0]
            )[0][1] + reopening_v_peak

            if right_most_index - left_most_index < 190:
                self.logger.info(f"blink segment too short: {right_most_index - left_most_index}. filtered.")
                continue

            left_most_indices.append(left_most_index)
            right_most_indices.append(right_most_index)
            closing_v_peaks_final.append(closing_v_peak)
            reopening_v_peaks_final.append(reopening_v_peak)

        return left_most_indices, right_most_indices, closing_v_peaks_final, reopening_v_peaks_final

    @abstractmethod
    def _init_labels(self, blink_ratio_inverse, blink_ind, blink_start, blink_end, closing_peak, reopening_peak) -> \
            tuple[int, int, int, int, dict]:
        """
        Determine blink phases from a blink.

        Returns:
            A tuple of indices for the closing, interphase, reopening, and non-blinking phases, and a dictionary of additional information.
        """
        pass

    @staticmethod
    def _prepare_blink_velocity(blink_ratio):
        blink_velocity = np.diff(blink_ratio, prepend=blink_ratio[0])

        gaussian_window = signal.windows.gaussian(101, std=31)
        normalized_gaussian_window = gaussian_window / np.sum(gaussian_window)
        blink_velocity = ndimage.convolve1d(blink_velocity, normalized_gaussian_window, axis=0)
        return blink_velocity

    def process(self, blink_ratio, non_blinking_mask=None, return_additional_info=False):
        """
        The main interface to generate blink labels from blink ratio.

        This code is a framework to be used by the subclasses. The subclasses should implement the _init_labels method.
        In this framework, velocity is used to locate each blink. Subclasses can use different methods to locate key
        points of the blinks.
        """
        numtaps = 101
        cutoff_f = 0.025
        blink_ratio = signal.filtfilt(
            signal.firwin(numtaps, cutoff_f), 1, x=blink_ratio
        )

        blink_ratio_inverse = 1 - blink_ratio

        blink_clusters = self.get_clusters_closed_interval(np.where(np.logical_not(non_blinking_mask))[0])

        blink_velocity = self._prepare_blink_velocity(blink_ratio)
        closing_v_peaks, reopening_v_peaks = self.find_and_match_peaks(blink_velocity, non_blinking_mask)

        init_labels = np.full_like(
            blink_ratio, fill_value=blink_defs.NOT_INITIALIZED, dtype=int
        )

        additional_infos = []
        for cluster in blink_clusters:
            self.logger.info(f"cluster {cluster[0]} - {cluster[1]}")
            self.logger.info("-" * 20)

            blink_starts, blink_ends, closing_peaks, reopening_peaks = self.get_blinks_from_cluster(
                blink_ratio_inverse, blink_velocity, cluster[0], cluster[1], closing_v_peaks, reopening_v_peaks,
            )
            for blink_ind in range(len(blink_starts)):
                (closing_start_index, interphase_start_index,
                 reopening_start_index, non_blinking_start_index, additional_info) = self._init_labels(
                    blink_ratio_inverse,
                    blink_ind,
                    blink_starts[blink_ind],
                    blink_ends[blink_ind],
                    closing_peaks[blink_ind],
                    reopening_peaks[blink_ind]
                )

                init_labels[closing_start_index] = blink_defs.CLOSING
                init_labels[interphase_start_index] = blink_defs.INTERPHASE
                init_labels[reopening_start_index] = blink_defs.REOPENING
                init_labels[non_blinking_start_index] = blink_defs.NON_BLINKING

                if return_additional_info:
                    additional_infos.append(additional_info)

        if return_additional_info:
            return init_labels, additional_infos
        else:
            return init_labels
