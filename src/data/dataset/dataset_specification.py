import json
from typing import Literal, Sequence, Optional, Union

import numpy as np
from scipy import signal, ndimage

from src.core import constants, validate_literal_args
from ..signal_processing import range_querying_fft, window_stats

# We define a processing protocol as a protocol on how to process the radar data.
# it is a sequence of steps that are applied to the radar data in order.
# each step is a component of the protocol.

# the following are the valid components that can be used in the protocol.
valid_components_list = [
    # initial frequency analysis methods
    "range-querying-fft",
    # filtering
    "low-pass-filtering",
    # diff
    "diff",
    # normalization
    "normalization",
]

# type aliases
ProcessingProtocolComponent = Literal[
    "range-querying-fft",
    "low-pass-filtering",
    "diff",
    "normalization",
]
ProcessingProtocol = Sequence[ProcessingProtocolComponent]

# different definition of eye openness (or blink parameters).
# See `data/README.md` for visualizations of these definitions.
BlinkParameter = Literal[
    "blink_ratio", "projected_blink_ratio", "eye_aspect_ratio", "projected_eye_aspect_ratio", "eye_occupancy"
]


class ProcessingConfig:
    """
    Configuration for processing the raw dataset.
    """
    range_resolution = 3.75  # cm
    n_antennas = 3

    @validate_literal_args
    def __init__(
            self,
            min_range=2.5,
            max_range=5.5,
            n_range_bins=10,
            stat_window_size=30,
            stat_hop=10,
            norm_stat: Literal["std", "max"] = "max",
    ):
        """
        Configuration for processing radar data.

        Args:
            min_range: Minimum range for range querying. In centimeters. Default: 2.5.
            max_range: Maximum range for range querying. In centimeters. Default: 5.5.
            n_range_bins: Number of range bins between min and max ranges. Default: 10.
            stat_window_size: Size of the statistical window. In seconds. Default: 30.
            stat_hop: Hop size for statistical windowing. In seconds. Default: 10.
            norm_stat: Normalization statistic to use. Default: "max". Supported: "std", "max".
        """
        # frequency domain analysis parameters (range/doppler beamforming).
        self.min_range = min_range
        self.max_range = max_range
        self.n_range_bins = n_range_bins

        # window used for calculating statistics
        self.stat_window_size = stat_window_size

        # statistics used for normalization
        self.stat_hop = stat_hop
        self.norm_stat = norm_stat

    @classmethod
    def load_from_json(cls, json_file):
        with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)
        return cls(**config_dict)

    def save_to_json(self, json_file):
        with open(json_file, 'w') as config_file:
            json.dump(self.__dict__, config_file)

    @property
    def beamforming_range_bin_indices(self):
        """
        Generate the range bin indices to query / beamform onto.
        """
        return np.linspace(self.min_range, self.max_range, self.n_range_bins) / self.range_resolution


class RadarDataProcessor:
    """
    Data processor for radar data in the raw dataset.
    """
    range_resolution = 3.75  # cm
    n_antennas = 3

    def __init__(
            self,
            processing_protocol: ProcessingProtocol = (
                    "range-querying-fft", "low-pass-filtering", "diff", "normalization"),
            processing_config: Union[str, dict, ProcessingConfig] = None,
    ):
        """
        A class that encapsulates the processing of radar data.

        Args:
            processing_protocol: a sequence of steps (``ProcessingProtocolComponent``) to apply to the radar data in order.
            processing_config: the configuration of parameters used in the processing. Can be a path to a JSON file, a dictionary, or an instance of ``ProcessingConfig``.
        """
        if processing_config is None:
            print("Processing configuration is not specified. Default will be used.")
            processing_config = ProcessingConfig()
        elif isinstance(processing_config, str):
            print(f"Loading processing configuration from {processing_config}.")
            processing_config = ProcessingConfig.load_from_json(processing_config)
        elif isinstance(processing_config, dict):
            processing_config = ProcessingConfig(**processing_config)

        self.processing_config = processing_config

        # frequency domain analysis parameters (range/doppler beamforming).
        self.min_range = processing_config.min_range
        self.max_range = processing_config.max_range
        self.n_range_bins = processing_config.n_range_bins

        # window used for calculating statistics
        self.stat_window_size = processing_config.stat_window_size

        # statistics used for normalization
        self.stat_hop = processing_config.stat_hop
        self.norm_stat = processing_config.norm_stat

        # processing protocol
        self._validate_processing_protocol(processing_protocol)
        self.processing_protocol = processing_protocol


    @staticmethod
    def _validate_processing_protocol(processing_protocol: ProcessingProtocol):
        """
        validate the provided protocol.
        """
        # rule 0: type checks
        #   0.1: is the protocol provided as a list?
        if not isinstance(processing_protocol, Sequence):
            raise TypeError("The processing protocol should be provided as a sequence (list or tuple).")
        #   0.2: are all components valid?
        if any(map(lambda x: x not in valid_components_list, processing_protocol)):
            raise ValueError("Invalid components in the list. Got {}, Supported: {}".format(
                list(filter(lambda x: x not in valid_components_list, processing_protocol)), valid_components_list
            ))

        # rule 1: only one differentiator operator should be specified (diff or normalized-diff).
        if len(list(filter(lambda c: "diff" in c, processing_protocol))) > 1:
            raise ValueError("Only one velocity processing technique can be specified. Got both.")

    def _apply_protocol_component(self, data, protocol_component):
        """
        apply the specified protocol component to the data.
        """
        if protocol_component == "range-querying-fft":
            data = self._range_querying_fft(data)
        elif protocol_component == "low-pass-filtering":
            data = self._low_pass_filtering(data)
        elif protocol_component == "diff":
            data = self._diff(data)
        elif protocol_component == "normalization":
            data = self._normalize(data)
        return data

    def _range_querying_fft(self, data):
        processed_data = range_querying_fft(data, self.processing_config.beamforming_range_bin_indices)
        return processed_data

    @staticmethod
    def _low_pass_filtering(data):
        b = signal.firwin(101, 30, pass_zero="lowpass", fs=constants.RADAR_FPS)
        processed_data = signal.filtfilt(b, 1, data, axis=0)
        return processed_data

    @staticmethod
    def _diff(data):
        velocity = np.diff(data, prepend=data[:1, ...], axis=0)
        return velocity

    def _normalize(
            self,
            data,
            fps: Optional[float] = round(constants.RADAR_FPS),
    ):
        n_dims = len(data.shape)
        # normalization stats
        stats = window_stats(
            np.mean(np.abs(data), axis=tuple(range(1, n_dims))),
            window_size=int(self.stat_window_size * fps),
            hop=int(self.stat_hop * fps),
            stat_func=self.norm_stat,
            apply_ema=True,
            return_full_size=True,
            lagged_stats=True
        )

        # keep original values where std is too small, otherwise apply normalization
        normalized_data = data / np.expand_dims(stats, tuple(range(1, n_dims)))
        return normalized_data

    def apply(self, data):
        """
        uniform interface to process the radar data given the specified protocol component.
        """
        for protocol_component in self.processing_protocol:
            data = self._apply_protocol_component(data, protocol_component)
        return data


class VisionDataProcessor:
    """
    Data processor for vision data in the raw dataset.
    """

    @staticmethod
    def _get_clusters(label_data, condition) -> list[tuple[int, int]]:
        """Identify clusters in the data based on the given condition."""
        indices = np.where(condition(label_data))[0]
        starts = indices[np.where(np.diff(indices, prepend=-2) > 1)[0]]
        # +1 to include the last element in the cluster
        ends = indices[np.where(np.diff(indices, append=label_data.shape[0] + 2) > 1)[0]] + 1
        return list(zip(starts, ends))

    @staticmethod
    def _process_blink_segment(blink_segment, lift_to=0.99):
        """Process a blink segment by lifting the lower half to the same level of the upper half."""
        min_index = np.argmin(blink_segment)
        min_value = blink_segment[min_index]

        start_value = blink_segment[0]
        end_value = blink_segment[-1]

        # Determine which half needs adjustment
        if start_value < end_value:
            lower_half = blink_segment[:min_index + 1]
            upper_half = blink_segment[min_index:]
            target_value = end_value
        else:
            lower_half = blink_segment[min_index:]
            upper_half = blink_segment[:min_index + 1]
            target_value = start_value

        # Rescale the lower half
        current_range = np.max(lower_half) - np.min(lower_half)
        target_range = target_value - min_value

        if current_range != 0:  # Avoid division by zero
            scale_factor = target_range / current_range
            lower_half_rescaled = min_value + (lower_half - min_value) * scale_factor
        else:
            lower_half_rescaled = lower_half  # If all values are the same, no rescaling needed

        # Combine the rescaled lower half with the upper half
        if start_value < end_value:
            balanced_curve = np.concatenate((lower_half_rescaled, upper_half[1:]))
        else:
            balanced_curve = np.concatenate((upper_half[:-1], lower_half_rescaled))

        # Calculate the scaling factor to bring the maximum to 0.99
        final_scale_factor = (lift_to - min_value) / (target_value - min_value)

        # Apply the scaling while preserving the minimum
        normalized_curve = min_value + (balanced_curve - min_value) * final_scale_factor

        return normalized_curve

    def normalize_vision_data(self, data, labels):
        """
        Normalize the vision data between 0 and 1.

        It removes effects from distance and perspective changes caused by head movements.
        """
        # smooth
        gaussian_window = signal.windows.gaussian(51, std=7)
        normalized_gaussian_window = gaussian_window / np.sum(gaussian_window)
        data = ndimage.convolve1d(data, normalized_gaussian_window, axis=0)

        max_distance = np.max(data)
        processed_data = np.copy(data)

        # remove trend in vision data caused by head movements
        #   non-blink segments
        non_blink_clusters = self._get_clusters(labels, lambda x: x <= 0)
        for start, end in non_blink_clusters:
            processed_data[start:end] = max_distance
        #   blinks
        blink_clusters = self._get_clusters(labels, lambda x: x > 0)
        for start, end in blink_clusters:
            blink_segment_distance = processed_data[start:end]
            processed_data[start:end] = self._process_blink_segment(blink_segment_distance, max_distance)

        # smooth again
        b, a = signal.butter(3, 0.1, btype='low', output='ba')
        processed_data = signal.filtfilt(b, a, processed_data)

        # normalize between 0 and 1
        low = 0.01
        high = 0.99
        min_val = np.min(processed_data)
        max_val = np.quantile(processed_data, 0.99)
        processed_data = (high - low) * (processed_data - min_val) / (max_val - min_val) + low

        # some experiments have denied regions where labels are incorrect
        # this step removes unnatural values.
        processed_data[processed_data >= 1] = 0.99
        return processed_data
