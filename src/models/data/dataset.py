import abc
import json
import math
import os
import re
import tempfile
import warnings
from abc import ABC
from pathlib import Path
from typing import Sequence, Iterator, Literal, Optional, cast, Union

import numpy as np
import tensorflow as tf

from src.core import project_files, blink_defs
from .sampler import sampler_factory, Sampler, BlinkSampler
from ..config import BaseDatasetConfig, BlinkDatasetConfig


class SingleDataset(ABC):
    def __init__(
            self,
            data_folder: Union[str, Path],
            experiment_name: str,
            config: BaseDatasetConfig,
            cache_folder: Optional[Union[str, Path]] = None
    ):
        """
        A dataset representing the data from one single experiment.
        """
        # path management
        self.data_folder = Path(data_folder)
        self.experiment_name = experiment_name

        # configuration
        self.config = config

        # filepaths
        self.experiment_folder = self.data_folder / self.experiment_name
        self.dataset_processing_protocol_path = self.experiment_folder / project_files.dataset_processing_protocol_filename
        self.input_path = self.experiment_folder / project_files.model_input_filename
        self.label_path = self.experiment_folder / project_files.model_label_filename

        # sampler
        self.sampler: Sampler = sampler_factory(config)

        # for caching dataset
        self.cache_folder = None if cache_folder is None else Path(cache_folder)
        self.cached_model_input_path = None
        self.cached_model_output_path = None
        if self.cache_folder is not None:
            self.cached_model_input_path = self.cache_folder / f"{self.experiment_name}_cached_input.npy"
            self.cached_model_output_path = self.cache_folder / f"{self.experiment_name}_cached_output.npy"

    @property
    def current_samples(self):
        return self.sampler.current_samples

    def _validate_dataset_processing_protocol(self):
        if not self.dataset_processing_protocol_path.exists():
            raise FileNotFoundError(
                f"Can not find the processing protocol file for {self.experiment_name}."
            )

        with open(self.dataset_processing_protocol_path, 'r') as file:
            recorded_dataset_processing_protocol = [line.strip() for line in file]

        error_message = (
            f"Expected dataset processing protocol does not match with the current processing protocol. "
            f"Expected {self.config.dataset_processing_protocol}, got {recorded_dataset_processing_protocol}. "
            f"Try construct the dataset again.")
        assert len(self.config.dataset_processing_protocol) == len(recorded_dataset_processing_protocol), error_message

        for expected, got in zip(self.config.dataset_processing_protocol, recorded_dataset_processing_protocol):
            if not expected == got:
                raise ValueError(error_message)

    def _preprocessing_and_get_data(self, crop) -> tuple[np.ndarray, np.ndarray]:
        self._validate_dataset_processing_protocol()

        input_data = np.load(self.input_path)
        with np.load(self.label_path) as label_data_file:
            label_data = label_data_file[self.config.event_detector]

        # for train, validation and test split
        begin_index, end_index = self.crop_to_start_end_index(crop, label_data)
        # print(f"Begin: {begin_index}, End: {end_index}")
        input_data = input_data[begin_index:end_index]
        label_data = label_data[begin_index:end_index]

        for processing_protocol in self.config.onsite_processing_protocol:
            if processing_protocol == "mean-beamforming":
                input_data = np.mean(input_data, axis=-1, keepdims=True)
            elif "select-bins" in processing_protocol:
                n_bins = int(processing_protocol.split("-")[-1])
                sample_bin_indices = np.linspace(0, input_data.shape[1], n_bins, endpoint=False).astype(int)
                input_data = input_data[:, sample_bin_indices]

        if self.config.modality == "magnitude-angle":
            # use both magnitude and angle as input
            magnitude = np.abs(input_data)
            angle = np.angle(input_data)
            processed_input_data = np.concatenate((magnitude, angle), axis=-1)
        elif self.config.modality == "magnitude":
            # only input magnitude of the signal
            magnitude = np.abs(input_data)
            processed_input_data = magnitude
        elif self.config.modality == "real-imaginary":
            # input data as real and imaginary parts
            processed_input_data = np.concatenate((np.real(input_data), np.imag(input_data)), axis=-1)
        else:
            raise ValueError(f"Unknown modality for pre-processing radar data {self.config.modality}.")

        return processed_input_data, label_data

    def _get_output_signature(self, temporal_dimension):
        sample_input = self._preprocessing_and_get_data(crop=(0, 0.1))[0]
        input_data_shape = sample_input.shape

        input_signature = tf.TensorSpec(
            shape=(temporal_dimension,) + tuple(input_data_shape[1:]),
            dtype=tf.float64
        )
        label_signature = tf.TensorSpec(
            shape=(temporal_dimension,),
            dtype=tf.float64
        )

        return input_signature, label_signature

    def crop_to_start_end_index(self, crop, labels: np.ndarray = None) -> tuple[int, int]:
        if labels is None:
            with np.load(self.label_path) as label_data_file:
                labels = label_data_file[self.config.event_detector]

        temporal_length = labels.shape[0]
        begin_index = max(0, math.floor(temporal_length * crop[0]))
        end_index = min(temporal_length, math.floor(temporal_length * crop[1]))

        # additional check: make sure no non-complete samples are included
        # both begin_index and end_index are adjusted forwards, to avoid overlapping cropping.
        begin_label = labels[begin_index]
        if begin_label != blink_defs.NON_BLINKING:
            if begin_index == 0:
                pass
            else:
                new_begin_indices = np.where(labels[:begin_index] == blink_defs.NON_BLINKING)[0]
                if len(new_begin_indices) > 0:
                    begin_index = max(0, new_begin_indices[-1] - 100)
                else:
                    warnings.warn("Cannot find a non-blinking event before the beginning of the crop.")

        end_label = labels[end_index - 1]
        if end_label != blink_defs.NON_BLINKING:
            if end_index == temporal_length:
                pass
            else:
                new_end_indices = np.where(labels[:end_index] == blink_defs.NON_BLINKING)[0]
                if len(new_end_indices) > 0:
                    end_index = new_end_indices[-1]
                else:
                    warnings.warn("Cannot find a non-blinking event after the end of the crop.")

        assert begin_index < end_index, "Begin index should be smaller than end index."

        return begin_index, end_index

    @abc.abstractmethod
    def get_output_signature(self):
        pass

    @abc.abstractmethod
    def sample_subset(self, crop=(0, 1), n_samples=4, deny_list: Sequence[tuple[int, int]] = None, **kwargs) -> \
            Sequence[np.ndarray]:
        pass

    @abc.abstractmethod
    def get_sampled_indices(self, crop=(0, 1), deny_list: Sequence[tuple[int, int]] = None):
        pass

    @abc.abstractmethod
    def as_dataset(self, crop=(0, 1), deny_list: Sequence[tuple[int, int]] = None):
        pass


class SingleBlinkExperimentDataset(SingleDataset):
    def __init__(
            self,
            data_folder: Union[str, Path],
            experiment_name,
            config: BlinkDatasetConfig,
            cache_folder: Optional[Union[str, Path]] = None
    ):
        """
        a dataset including blink signals
        """
        super().__init__(data_folder, experiment_name, config, cache_folder)
        # this is duplicated. added to make static type checking valid
        self.config = config
        self.sampler = cast(BlinkSampler, self.sampler)

        # target curve path.
        self.target_path = self.experiment_folder / project_files.model_target_ratio_filename

        # additional information about the sample
        self.output_signature = self.get_output_signature()

        # for caching
        self.cached_additional_info_path = None
        if self.cache_folder is not None:
            self.cached_additional_info_path = self.cache_folder / f"{self.experiment_name}_cached_additional_info.npy"

    def _get_video_label(self, crop) -> np.ndarray:
        with np.load(self.label_path) as label_data_file:
            video_label_data = label_data_file.get("video", None)
            if video_label_data is None:
                raise ValueError("No video label data found in the label file. Is the dataset constructed correctly?")

        begin_index, end_index = self.crop_to_start_end_index(crop)
        valid_label = video_label_data[begin_index:end_index]

        return valid_label

    def _get_target_data(self, crop, processing_protocol: Sequence[Literal["abs", "shift", "scale"]]) -> np.ndarray:
        # this will only be called by BlinkDatasets.
        target_data = np.load(self.target_path)[self.config.curve_name]

        if target_data.shape[0] == 0:
            raise ValueError(f"Target data (vision curve) is empty for {self.experiment_name}.")

        # crop
        begin_index, end_index = self.crop_to_start_end_index(crop)
        valid_target = target_data[begin_index:end_index]

        # processing
        if "abs" in processing_protocol:
            valid_target = np.abs(valid_target)
        if "shift" in processing_protocol:
            min_val = np.min(valid_target)
            valid_target = valid_target - min_val
        if "scale" in processing_protocol:
            valid_target = valid_target * 0.75

        return valid_target

    def sample_subset(self, crop=(0, 1), n_samples=4, deny_list: Sequence[tuple[int, int]] = None, **kwargs) -> \
            Sequence[np.ndarray]:
        input_data, label_data = self._preprocessing_and_get_data(crop)

        input_subsets = []
        label_subsets = []

        include_additional_info = kwargs.get("include_additional_info", False)

        sampling_types = kwargs.get("sampling_types", None)
        if sampling_types is None:
            sampling_types = tuple(self.config.sample_non_blinks) + ("blink",)

        for sampling_type in tuple(sampling_types):
            self.sampler.sampling_types = (sampling_type,)
            n_samples_type = kwargs.get(f"{sampling_type}_n_samples", n_samples)

            sampled_indices = self.sampler.sample_indices(
                label_data,
                deny_list=deny_list,
                ignore_shuffle=True,
                video_label_data=self._get_video_label(crop)
            )
            if len(sampled_indices) == 0:
                continue

            sampled_input, (sampled_label, additional_info) = self.sampler.sample_dataset(
                input_data,
                label_data,
                target_data=self._get_target_data(crop, self.config.curve_processing_protocol),
                sampled_indices=sampled_indices
            )

            max_n_samples = sampled_input.shape[0]
            if max_n_samples < n_samples_type:
                raise ValueError(
                    f"Cannot provide {n_samples_type} for {sampling_type}, as the max # is {max_n_samples}.")
            subset_sampled = np.round(np.linspace(0, max_n_samples - 1, n_samples_type)).astype(int)

            input_subsets.append(sampled_input[subset_sampled])

            if include_additional_info:
                label_subsets.append(np.concatenate(
                    [sampled_label[subset_sampled], additional_info[subset_sampled]],
                    axis=1)
                )
            else:
                label_subsets.append(sampled_label[subset_sampled])

        # reset sampler sampling types
        self.sampler.sampling_types = tuple(self.config.sample_non_blinks) + ("blink",)

        if len(input_subsets) == 0:
            return np.array([]), np.array([])
        else:
            return np.concatenate(input_subsets, axis=0), np.concatenate(label_subsets, axis=0)

    def get_output_signature(self):
        input_signature, label_signature = self._get_output_signature(
            temporal_dimension=self.config.uniform_length,
        )
        # additional information about the sample (resize ratio or valid length) is added.
        label_signature = tf.TensorSpec(
            shape=(label_signature.shape[0] + 1,) + label_signature.shape[1:],
            dtype=label_signature.dtype
        )
        return input_signature, label_signature

    def get_sampled_indices(self, crop=(0, 1), deny_list: Sequence[tuple[int, int]] = None):
        _, label_data = self._preprocessing_and_get_data(crop)
        sampled_indices = self.sampler.sample_indices(
            label_data,
            deny_list=deny_list,
            video_label_data=self._get_video_label(crop)
        )
        return sampled_indices

    def as_dataset(self, crop=(0, 1), deny_list: Sequence[tuple[int, int]] = None):
        if self.cached_model_input_path is not None and self.cached_model_output_path is not None and \
                self.cached_model_input_path.exists() and self.cached_model_output_path.exists():
            sampled_input = np.load(self.cached_model_input_path)
            sampled_label = np.load(self.cached_model_output_path)
            additional_info = np.load(self.cached_additional_info_path)
        else:
            input_data, label_data = self._preprocessing_and_get_data(crop)

            # sample from the entire sequence
            sampled_input, (sampled_label, additional_info) = self.sampler.sample_dataset(
                input_data,
                label_data,
                target_data=self._get_target_data(crop, self.config.curve_processing_protocol),
                deny_list=deny_list,
                video_label_data=self._get_video_label(crop)
            )

            # cache the dataset
            if self.cached_model_input_path is not None and self.cached_model_output_path is not None:
                np.save(self.cached_model_input_path, sampled_input)
                np.save(self.cached_model_output_path, sampled_label)
                np.save(self.cached_additional_info_path, additional_info)

        # construct the dataset
        n_expand_dims = sampled_label.ndim - 1
        additional_info = np.expand_dims(additional_info, axis=tuple(range(1, n_expand_dims)))
        sampled_label = np.concatenate([sampled_label, additional_info], axis=1)

        dataset = tf.data.Dataset.from_tensor_slices((sampled_input, sampled_label))

        return dataset


class AllExperimentDataset:
    exp_folder_pattern = r'^exp-\d{8}-\d{6}$'

    def __init__(self, data_folder, config: BaseDatasetConfig, exp_subset: Sequence[str] = None, cache: bool = True):
        """
        the dataset that represents all experiment data
        """
        self.data_folder = data_folder

        # read in valid experiments
        exp_folders = os.listdir(self.data_folder)
        exp_folders = list(filter(
            lambda f: re.match(self.exp_folder_pattern, f),
            exp_folders
        ))
        if exp_subset is not None and len(exp_subset) > 0:
            exp_folders = list(filter(
                lambda f: any((substring in f) for substring in exp_subset),
                exp_folders
            ))
        self.exp_folders = sorted(exp_folders)
        self.n_exps = len(self.exp_folders)

        # manage deny list
        rf_deny_list_path = os.path.join(data_folder, project_files.rf_deny_list_filename)
        self.rf_deny_list = {}
        if os.path.exists(rf_deny_list_path):
            with open(rf_deny_list_path) as f:
                self.rf_deny_list = json.load(f)

        # other properties
        self.config = config
        self.output_signature = self.get_output_signature()

        # for caching sample indices
        if cache:
            self.temp_dir = tempfile.mkdtemp(dir=self.data_folder)
        else:
            self.temp_dir = None

    def cleanup(self):
        import shutil
        print(f"Deleting temp dir {self.temp_dir}")
        if self.temp_dir is not None:
            try:
                shutil.rmtree(self.temp_dir)
                print("Temp dir deleted.")
            except FileNotFoundError:
                print("Temp dir not found. Did you run cleanup before?")
        else:
            print("Caching is not toggled. No clean up completed.")

    def _get_single_exp_sampled_indices(self, exp_folder, crop=(0, 1)) -> np.ndarray:
        indices = single_dataset_factory(self.data_folder, exp_folder, self.config).get_sampled_indices(
            crop=crop,
            deny_list=self.rf_deny_list.get(exp_folder, [])
        )
        return indices

    def _get_single_exp_data_generator(self, crop=(0, 1)):
        for exp_folder in self.exp_folders:
            current_ds = single_dataset_factory(
                self.data_folder,
                exp_folder,
                self.config,
                self.temp_dir
            ).as_dataset(
                crop=crop,
                deny_list=self.rf_deny_list.get(exp_folder, [])
            )
            for x, y in current_ds:
                yield x, y

    def sample_subset(self, crop=(0, 1), n_samples=4, **kwargs) -> Sequence[np.ndarray]:
        subset = []
        # count = 0
        for exp_folder in self.exp_folders:
            current_ds = single_dataset_factory(self.data_folder, exp_folder, self.config)
            subset_element = current_ds.sample_subset(
                crop=crop,
                n_samples=n_samples,
                deny_list=self.rf_deny_list.get(exp_folder, []),
                **kwargs
            )

            # count += subset_element[0].shape[0]
            # print(f"{exp_folder}: {subset_element[0].shape}, {subset_element[1].shape}. count: {count}")

            if len(subset) == 0:
                n_elements = len(subset_element)
                for i in range(n_elements):
                    subset.append([subset_element[i]])
            else:
                for e_ind, e in enumerate(subset_element):
                    subset[e_ind].append(e)

        subset = [np.concatenate(e_list, axis=0) for e_list in subset]

        return subset

    def get_output_signature(self):
        if len(self.exp_folders) == 0:
            raise ValueError(f"No experiment found in the provided folder {self.data_folder}.")

        return single_dataset_factory(
            self.data_folder, self.exp_folders[0], self.config
        ).get_output_signature()

    def get_sampled_indices(self, crop=(0, 1)):
        if self.config.shuffle:
            raise ValueError("Requesting sampled indices with shuffle flag on. Please unset shuffle.")
        for exp_folder in self.exp_folders:
            yield self._get_single_exp_sampled_indices(exp_folder, crop)

    def as_dataset(self, crop=(0, 1), return_dataset_directly=False):
        dataset = tf.data.Dataset.from_generator(
            lambda: self._get_single_exp_data_generator(crop=crop),
            output_signature=self.output_signature
        )
        if return_dataset_directly:
            return dataset
        else:
            return dataset.batch(self.config.batch_size)

    def as_per_experiment_dataset_list(self, crop) -> Iterator[SingleDataset]:
        for exp_folder in self.exp_folders:
            yield single_dataset_factory(
                self.data_folder,
                exp_folder,
                self.config,
                self.temp_dir
            ).as_dataset(
                crop=crop,
                deny_list=self.rf_deny_list.get(exp_folder, [])
            )


def single_dataset_factory(
        data_folder: Union[str, Path],
        experiment_name: str,
        config: BaseDatasetConfig,
        cache_folder: Optional[Union[str, Path]] = None
) -> SingleDataset:
    if config.sampler_type == "blink":
        return SingleBlinkExperimentDataset(
            data_folder,
            experiment_name,
            cast(BlinkDatasetConfig, config),
            cache_folder
        )
    else:
        raise ValueError(f"Unknown sampler type {config.sampler_type}.")
