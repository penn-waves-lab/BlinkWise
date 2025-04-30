from abc import abstractmethod
from pathlib import Path
from typing import Union, Literal, cast

import numpy as np

from src.core import validate_literal_args
from src.models.data import single_dataset_factory, SingleBlinkExperimentDataset
from ..utils import BaseProcessor


class CurveReconstructor(BaseProcessor):
    def __init__(
            self,
            exp_config_path: Union[str, Path],
            checkpoint_path: Union[str, Path],
            output_folder: Union[str, Path] = None,
            dataset_folder: Union[str, Path] = None,
            verbose: bool = True
    ):
        """
        Reconstruction of the openness curve using the trained model.

        Args:
            exp_config_path: Path to the experiment configuration file.
            checkpoint_path: Path to the model checkpoint.
            output_folder: Path to the output folder. If not provided, it will be read from the experiment config.
            dataset_folder: Path to the dataset folder. If not provided, it will be read from the experiment config.
            verbose: Whether to print logs.
        """
        self.checkpoint_path = Path(checkpoint_path)

        super().__init__(
            name=f"curve_reconstruction_{self.checkpoint_path.suffix[1:]}",
            exp_config_path=exp_config_path,
            output_folder=output_folder,
            verbose=verbose
        )

        self.dataset_folder = self.exp_config.data_folder if dataset_folder is None else Path(dataset_folder)
        self.model = self._load_model()

        for k, v in self.exp_config.to_dict().items():
            self.logger.info(f"{k}: {v}")

        self._rng = np.random.default_rng(seed=113)

    @abstractmethod
    def _load_model(self):
        """
        Load the model from the checkpoint. This method should be implemented by the subclass that uses different types of models.
        """
        pass

    @validate_literal_args
    def process(
            self,
            exp_name: str,
            crop_type: Literal["train", "validation"] = "validation",
            post_processing: bool = True,
    ) -> np.ndarray[float]:
        """
        Reconstruct the openness curve using the trained model.

        Args:
            exp_name: Name of the experiment session.
            crop_type: Type of the crop of data. Either "train" or "validation".
            post_processing: Whether to perform post-processing on the reconstructed curve. See `_merge_long_blinks`.

        Returns:
            The reconstructed openness curve.
        """
        # prepare the crop range
        crop = self.training_config.get_crop_range(crop_type)

        # overwrite certain properties of the dataset config
        ds_config = self.training_dataset_config.copy() if crop_type == "train" else self.validation_dataset_config.copy()
        ds_config.augmentation = ()
        ds_config.sample_non_blinks = ("event-no-expand",)
        ds_config.shuffle = False
        ds_config.random_margin = 0

        # prepare dataset for the specific experiment
        sed = single_dataset_factory(
            data_folder=self.dataset_folder,
            experiment_name=exp_name,
            config=ds_config,
        )
        sed = cast(SingleBlinkExperimentDataset, sed)
        crop_start_index, crop_end_index = sed.crop_to_start_end_index(crop)

        # load sampled segments
        sample_indices = sed.get_sampled_indices(crop=crop)

        sed_ds = sed.as_dataset(crop=crop)
        batch_size = sed_ds.cardinality().numpy()
        batched_dataset = sed_ds.batch(batch_size)
        sampled_input, sampled_label = next(iter(batched_dataset))

        # make prediction
        pred = self.model.predict(sed_ds.batch(batch_size), verbose=0)

        # processing reconstruction. the original curve is filled with 0.99
        # and the predicted values are filled in the sampled indices
        reconstructed_curve = np.full((crop_end_index - crop_start_index), 0.99)
        pred_flatten = []

        for sample_ind in range(batch_size):
            valid_len = int(sampled_label[sample_ind, -1])
            pred_flatten.append(np.squeeze(pred[sample_ind])[:valid_len])

        all_sampled_indices = np.concatenate(sample_indices)
        reconstructed_curve[all_sampled_indices] = np.concatenate(pred_flatten)

        # handling long blinks
        if post_processing:
            reconstructed_curve = self._merge_long_blinks(reconstructed_curve, sample_indices)

        return reconstructed_curve

    def _merge_long_blinks(self, reconstructed_curve, all_sampled_indices_list, incomplete_threshold=0.8):
        """
        Merge segmented long blinks in the reconstructed curve.

        When long blinks happen, the event proposal algorithm may break one blink into two segments (each for closing and reopening). This method merges the two segments into one.

        We detect incomplete blinks by checking the ending openness value of a segment. If the value is less than the threshold, we consider it as a potential incomplete blink. If the start openness value of the next segment is also less than the threshold, we merge the two segments.

        Args:
            reconstructed_curve: The reconstructed openness curve.
            all_sampled_indices_list: A list of indices of frames marked as events (can be blinks or other movements).
            incomplete_threshold: The threshold to determine if a segment is incomplete.

        Returns:
            The processed openness curve with merged long blinks.
        """
        all_sampled_indices_list = sorted(all_sampled_indices_list, key=lambda x: x[0])

        processed_curve = reconstructed_curve.copy()
        incomplete_blinks = []

        for seg_ind, sample_index in enumerate(all_sampled_indices_list):
            recon_segment = reconstructed_curve[sample_index]
            if recon_segment[-1] < incomplete_threshold:
                incomplete_blinks.append(seg_ind)

        for seg_ind in incomplete_blinks:
            if seg_ind == len(all_sampled_indices_list) - 1:
                self.logger.debug("The last segment is incomplete. Cannot merge.")
                continue

            # merge the current segment with the next segment
            current_segment_indices = all_sampled_indices_list[seg_ind]
            next_segment_indices = all_sampled_indices_list[seg_ind + 1]
            current_segment = reconstructed_curve[current_segment_indices]
            next_segment = reconstructed_curve[next_segment_indices]

            if next_segment[0] >= incomplete_threshold:
                self.logger.debug("Next segment is complete. Skipping...")
                continue

            self.logger.debug(f"Merged segment: {current_segment_indices[0]} - {next_segment_indices[-1]}")

            # use speed to fill the gap
            current_min_ind = np.argmin(current_segment) + current_segment_indices[0]
            next_min_ind = np.argmin(next_segment) + next_segment_indices[0]

            # gap_speed = np.random.randn((next_min_ind - current_min_ind)) * 5e-4

            gap_speed = self._rng.normal(loc=0.0, scale=5e-4, size=(next_min_ind - current_min_ind))

            processed_curve[current_min_ind:next_min_ind] = np.clip(
                np.cumsum(gap_speed) + current_segment[current_min_ind - current_segment_indices[0]],
                0.01, 0.99
            )

        return processed_curve
