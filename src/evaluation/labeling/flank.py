import math
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .base import AutoLabeler


class FlankAutoLabeler(AutoLabeler):
    """
    Auto labeler that uses flanks to determine the blink phases.

    See Also:
        Philipp P Caffier, Udo Erdmann, and Peter Ullsperger. 2003. Experimental evaluation of eye-blink parameters as a drowsiness measure. European journal of applied physiology 89 (2003), 319–325.
    """
    def __init__(self,
                 exp_config_path: Optional[Union[str, Path]] = None,
                 # needed by flank-based labeler
                 interphase_percentage=0.95, blink_percentage=0., flank_high_percentage=0.95, flank_low_percentage=0.5,
                 # needed by AutoLabeler
                 peak_quantile=0.85, ptp_quantile=0.95, closing_wlen=0.3, reopening_wlen=0.5, quantile_window=30,
                 fps=480, min_peak_distance=0.2,
                 output_folder: Optional[Union[str, Path]] = None,
                 verbose=True):
        """
        Initialize the FlankAutoLabeler.

        Args:
            exp_config_path: Path to the experiment configuration file.
            interphase_percentage: Percentage of the peak inverse openness as the minimum height for interphase phases.
            blink_percentage: Percentage of the peak inverse openness as the horizontal that intersects with the flanks to determine blink starts and ends.
            flank_high_percentage: Percentage of the peak inverse openness that determines the higher point to fit of a flank.
            flank_low_percentage: Percentage of the peak inverse openness that determines the lower point to fit of a flank.
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
        super().__init__(
            exp_config_path=exp_config_path,
            peak_quantile=peak_quantile,
            ptp_quantile=ptp_quantile,
            closing_wlen=closing_wlen,
            reopening_wlen=reopening_wlen,
            quantile_window=quantile_window,
            fps=fps,
            min_peak_distance=min_peak_distance,
            output_folder=output_folder,
            verbose=verbose
        )

        self.interphase_percentage = interphase_percentage
        self.blink_percentage = blink_percentage
        self.flank_high_percentage = flank_high_percentage
        self.flank_low_percentage = flank_low_percentage

    @staticmethod
    def _is_point_same_event(signal_values, point, peak, threshold=0.05) -> bool:
        """
        Check if the point is in the same blink event as the peak.

        If the point to check is way too small compared to the peak, it is considered as a different event.

        Args:
            signal_values: Inverse openness curve.
            point: Index of the point to be checked.
            peak: The peak index.
            threshold: Minimum threshold for the point to be considered in the same event.

        Returns:
            Whether the point is in the same event as the peak.
        """
        if point < peak:
            segment = signal_values[point:peak]
        elif point > peak:
            segment = signal_values[peak:point]
        else:
            return True
        return np.min(segment) >= threshold * signal_values[peak]

    def get_clusters_in_same_event(self, signal_values, indices, peak, threshold=0.05):
        clusters = self.get_clusters_closed_interval(indices)
        return [cluster for cluster in clusters if
                self._is_point_same_event(signal_values, cluster[1], peak, threshold)]

    def _init_labels(self, blink_ratio_inverse, blink_ind, blink_start, blink_end, closing_peak, reopening_peak):
        """
        Determine blink phases based on linear regression on both flanks.
        """
        peak = np.argmax(blink_ratio_inverse[closing_peak:reopening_peak]) + closing_peak
        peak_height = blink_ratio_inverse[peak]

        flank_low_height = self.flank_low_percentage * peak_height
        flank_high_height = self.flank_high_percentage * peak_height

        left_side = np.arange(blink_start, peak)
        right_side = np.arange(peak, blink_end)

        # decide where flanks start and end
        #     -----------------------
        #     for left side: flank starts from the first point higher than low thred to the that higher than high thred
        #     -----------------------
        left_side_higher_than_flank_low = np.where(blink_ratio_inverse[left_side] > flank_low_height)[0]
        if left_side_higher_than_flank_low.shape[0] == 0:
            self.logger.info(f"No left side value is higher than flank low height for peak # {blink_ind} @ {peak}")
            left_flank_start = np.where(blink_ratio_inverse[left_side] <= flank_low_height)[0][-1] + blink_start
        else:
            # we choose the farthest cluster to the peak within the same event
            left_flank_starts = left_side_higher_than_flank_low + blink_start
            left_flank_start = self.get_clusters_in_same_event(
                blink_ratio_inverse, left_flank_starts, peak, threshold=self.blink_percentage
            )[0][0]

        left_side_higher_than_flank_high = np.where(blink_ratio_inverse[left_side] > flank_high_height)[0]
        if left_side_higher_than_flank_high.shape[0] == 0:
            self.logger.info(f"No left side value is higher than flank high height for peak # {blink_ind} @ {peak}")
            left_flank_end = peak
        else:
            # we choose the farthest cluster to the peak within the same event
            left_flank_ends = left_side_higher_than_flank_high + blink_start
            left_flank_end = self.get_clusters_in_same_event(
                blink_ratio_inverse, left_flank_ends, peak, threshold=self.blink_percentage
            )[0][0]

        #     fit left flanks
        self.logger.debug(f"Left flank before: {left_flank_start} - {left_flank_end}")
        if left_flank_end <= left_flank_start:
            self.logger.info("Left flank end is less than or equal to start: {} <= {}".format(
                left_flank_end, left_flank_start
            ))
            left_flank_start = np.where(blink_ratio_inverse[left_side] <= flank_low_height)[0][-1] + blink_start
        self.logger.debug(f"Left flank after: {left_flank_start} - {left_flank_end}")

        left_flank_a, left_flank_b = np.polyfit(
            np.arange(left_flank_start, left_flank_end + 1),
            blink_ratio_inverse[left_flank_start: left_flank_end + 1],
            1,
        )

        #     -----------------------
        #     for right side: flank starts the higher point to the lower point
        #     -----------------------
        right_side_higher_than_flank_high = np.where(blink_ratio_inverse[right_side] > flank_high_height)[0]
        if right_side_higher_than_flank_high.shape[0] == 0:
            self.logger.info(f"No right side value is higher than flank high height for peak # {blink_ind} @ {peak}")
            right_flank_start = peak
        else:
            # we choose the farthest cluster to the peak within the same event
            right_flank_starts = right_side_higher_than_flank_high + peak
            right_flank_start = self.get_clusters_in_same_event(
                blink_ratio_inverse, right_flank_starts, peak, threshold=self.blink_percentage
            )[-1][-1]

        right_side_higher_than_flank_low = np.where(blink_ratio_inverse[right_side] > flank_low_height)[0]
        if right_side_higher_than_flank_low.shape[0] == 0:
            self.logger.info(f"No right side value is higher than flank low height for peak # {blink_ind} @ {peak}")
            right_flank_end = np.where(blink_ratio_inverse[right_side] <= flank_low_height)[0][-1] + peak
        else:
            # we choose the farthest cluster to the peak within the same event
            right_flank_ends = right_side_higher_than_flank_low + peak
            right_flank_end = self.get_clusters_in_same_event(
                blink_ratio_inverse, right_flank_ends, peak, threshold=self.blink_percentage
            )[-1][-1]

        self.logger.debug(f"Right flank before: {right_flank_start} - {right_flank_end}")
        if right_flank_end <= right_flank_start:
            self.logger.info("Right flank end is less than or equal to start: {} <= {}".format(
                right_flank_end, right_flank_start
            ))
            right_flank_end = np.where(blink_ratio_inverse[right_side] <= flank_low_height)[0][0] + peak
        self.logger.debug(f"Right flank after: {right_flank_start} - {right_flank_end}")

        right_flank_a, right_flank_b = np.polyfit(
            np.arange(right_flank_start, right_flank_end + 1),
            blink_ratio_inverse[right_flank_start: right_flank_end + 1],
            1,
        )

        # calculate the intersection point
        interphase_height = self.interphase_percentage * peak_height
        blink_height = self.blink_percentage * peak_height

        closing_start_index = math.floor((blink_height - left_flank_b) / left_flank_a)
        if closing_start_index < blink_start or closing_start_index > peak:
            closing_start_index = blink_start

        # interphase_start_index = max(blink_start, math.floor((interphase_height - left_flank_b) / left_flank_a))
        # reopening_start_index = min(blink_end, math.ceil((interphase_height - right_flank_b) / right_flank_a))
        non_blinking_start_index = math.ceil((blink_height - right_flank_b) / right_flank_a)
        if non_blinking_start_index < peak or non_blinking_start_index > blink_end:
            non_blinking_start_index = blink_end

        # determine the start and end of interphase/reopening
        left_side_higher_than_interphase_height = np.where(blink_ratio_inverse[left_side] > interphase_height)[0]
        right_side_higher_than_interphase_height = np.where(blink_ratio_inverse[right_side] > interphase_height)[0]

        if left_side_higher_than_interphase_height.shape[0] == 0:
            self.logger.info(f"No left side value is higher than interphase height for peak # {blink_ind} @ {peak}")
            interphase_start_index = peak - 1
        else:
            interphase_start_index = left_side_higher_than_interphase_height[0] + blink_start

        if right_side_higher_than_interphase_height.shape[0] == 0:
            self.logger.info(f"No right side value is higher than interphase height for peak # {blink_ind} @ {peak}")
            reopening_start_index = peak + 1
        else:
            reopening_start_index = right_side_higher_than_interphase_height[-1] + peak

        additional_info = {
            "left_start": left_flank_start,
            "left_end": left_flank_end,
            "right_start": right_flank_start,
            "right_end": right_flank_end,
            "left_flank": (left_flank_a, left_flank_b),
            "right_flank": (right_flank_a, right_flank_b),
        }

        return closing_start_index, interphase_start_index, reopening_start_index, non_blinking_start_index, additional_info
