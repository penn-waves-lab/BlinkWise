import abc
from abc import ABC
from typing import Union, Sequence, Optional, cast

import numpy as np
from intervaltree import IntervalTree
from scipy.interpolate import interp1d

from src.core import blink_defs
from ..config import BaseDatasetConfig, BlinkDatasetConfig


class Sampler(ABC):
    def __init__(self, config: BaseDatasetConfig):
        # sample configuration
        self.shuffle = config.shuffle
        self.seed = config.seed
        self.rng = np.random.default_rng(self.seed)

    @property
    @abc.abstractmethod
    def current_samples(self) -> int:
        """
        the number of samples in the current dataset
        """
        return 0

    @abc.abstractmethod
    def sample_indices(
            self,
            label_data: np.ndarray,
            deny_list: Sequence[tuple[int, int]] = None,
            ignore_shuffle: bool = False,
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:
        """
        get indices that construct samples.
        np.ndarray stores uniform-length indices; Sequence[np.ndarray] stores varying lengths.
        """
        pass

    @abc.abstractmethod
    def sample_dataset(
            self,
            input_data: np.ndarray,
            label_data: np.ndarray,
            target_data: np.ndarray = None,
            sampled_indices=None,
            deny_list: Sequence[tuple[int, int]] = None
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:
        """
        get the dataset
        """
        pass


class BlinkSampler(Sampler):
    def __init__(self, config: BlinkDatasetConfig):
        super().__init__(config)

        self.uniform_length = config.uniform_length
        self.resizing_method = config.resizing_method
        self.random_margin = config.random_margin

        self.augmentation = config.augmentation
        self.augmentation_ratio = config.augmentation_ratio

        self.sampling_types = tuple(config.sample_non_blinks) + ("blink",)

        self._current_samples = 0

    @staticmethod
    def _build_interval_tree(deny_list: Sequence[tuple[int, int]]) -> IntervalTree:
        tree = IntervalTree()
        if deny_list is None:
            return tree

        for start, end in deny_list:
            tree[start:end + 1] = (start, end)
        return tree

    @staticmethod
    def _find_closest_before(tree: IntervalTree, point: int) -> Optional[int]:
        """Find the maximum end in intervals strictly before the point."""
        candidates = tree[None:point]
        if candidates:
            return max(c.end for c in candidates if c.end < point)
        return None

    @staticmethod
    def _find_closest_after(tree: IntervalTree, point: int) -> Optional[int]:
        """Find the minimum start in intervals strictly after the point."""
        candidates = tree[point:]
        if candidates:
            return min(c.begin for c in candidates if c.begin > point)
        return None

    @staticmethod
    def _get_clusters(label_data, condition) -> list[tuple[int, int]]:
        indices = np.where(condition(label_data))[0]
        starts = indices[np.where(np.diff(indices, prepend=-2) > 1)[0]]
        ends = indices[np.where(np.diff(indices, append=label_data.shape[0] + 2) > 1)[0]]
        assert starts.shape[0] == ends.shape[0], "Cluster starts do not equal cluster ends."
        return list(zip(starts, ends))

    @staticmethod
    def _process_event_cluster(start, end, max_length) -> list[tuple[int, int]]:
        if end - start + 1 < 256:
            new_start = max(0, start - (256 - (end - start + 1)) // 2)
            new_end = min(max_length - 1, new_start + 255)
            new_start = max(0, new_end - 255)  # Adjust start if end hit the boundary
            return [(new_start, new_end)]
        else:
            return [(s, min(max_length - 1, s + 255)) for s in range(start, end - 254, 16)]

    @staticmethod
    def _process_negative_cluster(start, end, max_length) -> list[tuple[int, int]]:
        if end - start + 1 < 128:
            return []
        elif 128 <= end - start + 1 <= 256:
            return [(start, end)]
        else:
            return [(s, min(max_length - 1, s + 255)) for s in range(start, end - 254, 64)]

    @property
    def current_samples(self):
        return self._current_samples

    def _get_augment_clusters(self, blink_clusters, video_label_data) -> list[tuple[int, int]]:
        augmented_clusters = []
        do_augmentation = self.rng.random(len(blink_clusters)) < self.augmentation_ratio

        for cluster_ind, (start, end) in enumerate(blink_clusters):
            if not do_augmentation[cluster_ind]:
                continue

            video_label = video_label_data[start:end + 1]
            # print(f"Augmenting cluster {start} -> {end} with video label {video_label}")

            if "left" in self.augmentation:
                reopening_locations = np.where(video_label == blink_defs.REOPENING)[0]
                if len(reopening_locations) > 1:
                    reopening_start = reopening_locations[0] + start

                    if reopening_start - 1 <= start:
                        continue

                    augmented_clusters.append((start, reopening_start - 1))
                else:
                    pass

            if "right" in self.augmentation:
                closing_locations = np.where(video_label == blink_defs.CLOSING)[0]
                if len(closing_locations) > 1:
                    closing_end = closing_locations[-1] + start

                    if closing_end + 1 >= end:
                        continue

                    augmented_clusters.append((closing_end + 1, end))
                else:
                    pass

        return augmented_clusters

    def sample_indices(
            self,
            label_data: np.ndarray,
            deny_list: Sequence[tuple[int, int]] = None,
            ignore_shuffle: bool = False,
            video_label_data: np.ndarray = None,
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:
        max_len = label_data.shape[0]
        tree = self._build_interval_tree(deny_list)
        sample_indices = []

        for sampling_type in self.sampling_types:
            if sampling_type == "blink":
                clusters = self._get_clusters(label_data, lambda x: x > 0)
                if len(self.augmentation) > 0:
                    if video_label_data is None:
                        raise ValueError(f"Video label data is not provided for augmentation {self.augmentation}.")
                    augmented_clusters = self._get_augment_clusters(clusters, video_label_data)
                    clusters.extend(augmented_clusters)
                    clusters = sorted(clusters)
            elif "event" in sampling_type:
                clusters = self._get_clusters(label_data, lambda x: x == blink_defs.NON_BLINK_EVENT)
                if "no-expand" not in sampling_type:
                    clusters = [c for start, end in clusters for c in self._process_event_cluster(start, end, max_len)]
            elif sampling_type == "negative":
                clusters = self._get_clusters(label_data, lambda x: x == blink_defs.NON_BLINKING)
                clusters = [c for start, end in clusters for c in self._process_negative_cluster(start, end, max_len)]
            else:
                print(f"Unknown non-blink sampling spec at {self.__class__.__name__}: {sampling_type}")
                continue

            for cluster_start, cluster_end in clusters:
                if not tree.is_empty():
                    # Check overlap with intervals defined in the deny list
                    overlapping = tree[cluster_start:cluster_end + 1]
                    if len(overlapping) > 0:
                        continue

                    # Limit margins from overlapping when necessary
                    closest_before = self._find_closest_before(tree, cluster_start) or 0
                    closest_after = self._find_closest_after(tree, cluster_end) or max_len
                    extend_margin_start = min(self.random_margin, cluster_start - closest_before)
                    extend_margin_end = min(self.random_margin, closest_after - (cluster_end + 1))
                else:
                    extend_margin_start = extend_margin_end = self.random_margin

                # Apply random margins
                sample_start = max(0, cluster_start - self.rng.integers(0, extend_margin_start, endpoint=True))
                sample_end = min(max_len, cluster_end + 1 + self.rng.integers(0, extend_margin_end, endpoint=True))
                sample_indices.append(np.arange(sample_start, sample_end))

        if not ignore_shuffle and self.shuffle:
            self.rng.shuffle(sample_indices)

        return sample_indices

    def sample_dataset(
            self,
            input_data: np.ndarray,
            label_data: np.ndarray,
            target_data: np.ndarray = None,
            sampled_indices=None,
            deny_list: Sequence[tuple[int, int]] = None,
            video_label_data: np.ndarray = None,
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:
        if sampled_indices is None:
            sampled_indices = self.sample_indices(label_data, deny_list, video_label_data=video_label_data)

        sampled_input = []
        sampled_label = []
        additional_info = []

        for sample_ind, one_sample_indices in enumerate(sampled_indices):
            # Ti x 10 x 6/3, or Ti x 10 x 10 x 6/3
            blink_input_data = input_data[one_sample_indices]
            # Ti
            blink_label_data = label_data[one_sample_indices]

            if self.resizing_method == "resize":
                # interpolate each blink to the same size along the time dimension.
                original_length = blink_input_data.shape[0]
                x = np.linspace(0, original_length - 1, original_length)
                new_x = np.linspace(0, original_length - 1, self.uniform_length)

                # interpolate input signal.
                # output shape: UL x ...
                interp_func_input = interp1d(x, blink_input_data, axis=0, kind='cubic')
                blink_input_data = interp_func_input(new_x)

                # interpolate label.
                # output shape: UL x 4 (categorical) or UL (continuous).
                if target_data is None:
                    raise ValueError("Target data is not provided for resizing method 'resize'.")
                else:
                    # target is specified. use continuous interpolation function.
                    blink_target_data = target_data[one_sample_indices]
                    interp_func_label = interp1d(x, blink_target_data, axis=0, kind='cubic')
                    blink_label_data = interp_func_label(new_x)

                resize_ratio = one_sample_indices.shape[0] / self.uniform_length

                additional_info.append([resize_ratio])
                sampled_input.append(blink_input_data)
                sampled_label.append(blink_label_data)

            elif self.resizing_method == "pad":
                # pad input signal.
                # Pad the signal if shorter than the uniform length, or truncate if longer.
                # output shape: UL x 4 (categorical) or UL (continuous).
                original_length = blink_input_data.shape[0]
                n_other_dims = blink_input_data.ndim - 1

                def pad_segment(input_segment, label_segment, segment_indices, valid_length):
                    """Helper function to pad a segment to uniform length"""
                    # Pad input
                    padded_input = np.pad(
                        input_segment,
                        pad_width=np.array([[0, self.uniform_length - valid_length]] + [[0, 0]] * n_other_dims),
                        mode="edge"
                    )

                    # Pad and process label
                    if target_data is None:
                        raise ValueError("Target data is not provided for resizing method 'pad'.")
                    else:
                        segment_target = target_data[segment_indices]
                        padded_label = np.pad(
                            segment_target,
                            pad_width=np.array([[0, self.uniform_length - valid_length]]),
                            mode="edge"
                        )

                    return padded_input, padded_label

                if original_length > self.uniform_length:
                    # truncate the signal since it is longer than the uniform length.
                    n_complete_chunks = original_length // self.uniform_length
                    remaining_length = original_length % self.uniform_length

                    for chunk_idx in range(n_complete_chunks):
                        start_idx = chunk_idx * self.uniform_length
                        end_idx = start_idx + self.uniform_length

                        # Add complete chunk
                        chunk_input = blink_input_data[start_idx:end_idx]
                        sampled_input.append(chunk_input)

                        if target_data is None:
                            raise ValueError("Target data is not provided for resizing method 'pad'.")
                        else:
                            chunk_target = target_data[one_sample_indices[start_idx:end_idx]]
                            chunk_label = chunk_target

                        sampled_label.append(chunk_label)
                        additional_info.append([self.uniform_length])

                    # Handle remaining data if any
                    if remaining_length > 0:
                        remaining_input = blink_input_data[-remaining_length:]
                        remaining_label = blink_label_data[-remaining_length:]
                        remaining_indices = one_sample_indices[-remaining_length:]

                        padded_input, padded_label = pad_segment(
                            remaining_input,
                            remaining_label,
                            remaining_indices,
                            remaining_length
                        )

                        sampled_input.append(padded_input)
                        sampled_label.append(padded_label)
                        additional_info.append([remaining_length])
                else:
                    # Handle case where original length is shorter than uniform length
                    padded_input, padded_label = pad_segment(
                        blink_input_data,
                        blink_label_data,
                        one_sample_indices,
                        original_length
                    )

                    sampled_input.append(padded_input)
                    sampled_label.append(padded_label)
                    additional_info.append([original_length])

        sampled_input = np.stack(sampled_input, axis=0)
        sampled_label = np.stack(sampled_label, axis=0)
        additional_info = np.stack(additional_info, axis=0)

        # setup current sample count
        self._current_samples = sampled_input.shape[0]

        return sampled_input, (sampled_label, additional_info)


def sampler_factory(config: BaseDatasetConfig) -> Sampler:
    # cast() for suppressing warning raised for mismatched types of configs.
    if config.sampler_type == "blink":
        return BlinkSampler(cast(BlinkDatasetConfig, config))
    else:
        raise ValueError(f"Unknown sampler type {config.sampler_type}.")
