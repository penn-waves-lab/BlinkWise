import json
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, Sequence, Literal, Optional

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix

from src.core import blink_defs, project_files, validate_literal_args
from src.models.data import single_dataset_factory
from ..utils import BaseProcessor
from .helpers import from_spiky_labels_to_continuous_labels

"""
The type alias for pairing of predicted and target segments.

The key is the index of the target segment, and the value is a list of indices of the predicted segments that are matched to the target segment.

-1 if used to indicate a dummy segment (i.e., not matched to any segment).
"""
Pairing = dict[int, list[int]]


class Segment(ABC):
    """
    A segment in the data stream from on experiment session representing a blink or a inter=blink interval.
    """

    @property
    @abstractmethod
    def start(self) -> int:
        pass

    @property
    @abstractmethod
    def end(self) -> int:
        pass


class BlinkSegment(Segment):
    """
    A blink segment in the data stream from an experiment session.
    """

    def __init__(
            self,
            closing_start,
            interphase_start,
            reopening_start,
            non_blinking_start,
            is_empty=False,
    ) -> None:
        super(BlinkSegment, self).__init__()

        self.closing_start = closing_start
        self.interphase_start = interphase_start
        self.reopening_start = reopening_start
        self.non_blinking_start = non_blinking_start
        self.is_empty = is_empty

    @property
    def closing_duration(self):
        if self.is_empty:
            return 0
        return self.interphase_start - self.closing_start

    @property
    def interphase_duration(self):
        if self.is_empty:
            return 0
        if np.isnan(self.reopening_start):
            return np.nan
        return self.reopening_start - self.interphase_start

    @property
    def reopening_duration(self):
        if self.is_empty:
            return 0
        return self.non_blinking_start - self.reopening_start

    @property
    def duration(self):
        if self.is_empty:
            return 0
        return self.non_blinking_start - self.closing_start

    @property
    def start(self):
        return self.closing_start

    @property
    def end(self):
        return self.non_blinking_start


class NonBlinkSegment(Segment):
    """
    A non-blink segment in the data stream from an experiment session.
    """

    def __init__(self, non_blinking_start, non_blinking_end) -> None:
        super(NonBlinkSegment, self).__init__()

        self.non_blinking_start = non_blinking_start
        self.non_blinking_end = non_blinking_end

    @property
    def non_blinking_duration(self):
        return self.non_blinking_end - self.non_blinking_start

    @property
    def start(self):
        return self.non_blinking_start

    @property
    def end(self):
        return self.non_blinking_end


class MetricEvaluator(BaseProcessor):
    def __init__(
            self,
            exp_config_path: Union[str, Path],
            dataset_folder: Union[str, Path] = None,
            output_folder: Union[str, Path] = None,
            verbose: bool = True
    ):
        """
        Evaluate metrics for openness curve prediction.

        The evaluation algorithm follows:
        1. we convert the labels from auto labelers into segments of blinks and non-blinks (predicted). Similarly, the target labels are also converted into segments (target).
        2. we pair blink segments and non-blink segments from the predicted and target labels.
        3. based on the pairing, we calculate the metrics (correlation, RMSE, MAE, event detection, and phase analysis).

        Args:
            exp_config_path: Path to the experiment configuration file.
            dataset_folder: Path to the dataset folder. If not provided, it will be read from the experiment config.
            output_folder: Path to the output folder. If not provided, it will be read from the experiment config.
            verbose: Whether to print logs.
        """
        super().__init__(
            name="metric_evaluation",
            exp_config_path=exp_config_path,
            output_folder=output_folder,
            verbose=verbose
        )

        self.dataset_folder = self.exp_config.data_folder if dataset_folder is None else Path(dataset_folder)

        self.val_crop = self.training_config.get_crop_range("validation")

        rf_deny_list_path = self.dataset_folder / project_files.rf_deny_list_filename
        if not rf_deny_list_path.exists():
            raise FileNotFoundError(f"RF deny list file not found: {rf_deny_list_path}")
        with open(rf_deny_list_path, "r") as f:
            self.rf_deny_list = json.load(f)

    @staticmethod
    def labels_to_blink_segments(spiky_labels) -> list[BlinkSegment]:
        """
        Convert spiky labels to blink segments. See ``helpers.py`` for more information on spiky labels.

        Args:
            spiky_labels: The blink phase labels in spiky form.

        Returns:
            A list of blink segments
        """
        blink_segments = []
        closing_start = np.nan
        interphase_start = np.nan
        reopening_start = np.nan
        non_blinking_start = np.nan

        labeled_locs = np.where(spiky_labels >= 0)[0]

        for loc in labeled_locs:
            label = spiky_labels[loc]
            if label == blink_defs.CLOSING:
                closing_start = loc
            elif label == blink_defs.INTERPHASE:
                interphase_start = loc
            elif label == blink_defs.REOPENING:
                reopening_start = loc
            elif label == blink_defs.NON_BLINKING:
                non_blinking_start = loc

                blink_segments.append(
                    BlinkSegment(
                        closing_start, interphase_start, reopening_start, non_blinking_start
                    )
                )
                non_blinking_start = np.nan
                closing_start = np.nan
                interphase_start = np.nan
                reopening_start = np.nan

        return blink_segments

    @staticmethod
    def labels_to_non_blink_segments(spiky_labels) -> list[NonBlinkSegment]:
        """
        Convert spiky labels to non-blink segments. See ``helpers.py`` for more information on spiky labels.

        Args:
            spiky_labels: The blink phase labels in spiky form.

        Returns:
            A list of non-blink segments
        """
        non_blink_segments = []
        non_blinking_locs = np.where(spiky_labels == blink_defs.NON_BLINKING)[0]
        closing_locs = np.where(spiky_labels == blink_defs.CLOSING)[0]

        assert non_blinking_locs.shape[0] == closing_locs.shape[
            0], f"Non-blinking and closing mismatch. Non-blinking: {non_blinking_locs.shape[0]} vs closing: {closing_locs.shape[0]}"

        for nb, c in zip(non_blinking_locs[:-1], closing_locs[1:]):
            non_blink_segments.append(NonBlinkSegment(nb, c))

        return non_blink_segments

    def pair_segments(
            self, pred_segments: list[Segment], target_segments: list[Segment]
    ) -> Pairing:
        """
        Pair predicted and target segments.

        Args:
            pred_segments: A list of predicted segments.
            target_segments: A list of target segments.

        Returns:
            Pairing of predicted and target segments.
        """
        pred_starts = np.array([bs.start for bs in pred_segments])
        pred_ends = np.array([bs.end for bs in pred_segments])

        target_starts = np.array([bs.start for bs in target_segments])
        target_ends = np.array([bs.end for bs in target_segments])

        # if segments overlap, they are considered as matched
        condition1 = pred_starts[:, None] < target_ends[None, :]  # Shape (len(L1), len(L2))
        condition2 = pred_ends[:, None] > target_starts[None, :]  # Shape (len(L1), len(L2))
        pairing_mask = condition1 & condition2

        # track whether all predictions are matched
        is_pred_matched = np.zeros(pred_starts.shape[0], dtype=bool)

        # start pairing
        pairing = {}
        for target_ind in range(target_starts.shape[0]):
            pred_inds = np.where(pairing_mask[:, target_ind])[0]
            if pred_inds.shape[0] == 0:
                self.logger.debug(f"Target {target_ind} is not matched to any prediction.")
                pairing[target_ind] = [-1]
            else:
                pairing[target_ind] = pred_inds

                pred_ind = pred_inds[0]
                is_pred_matched[pred_ind] = True

        unmatched_pred_inds = np.where(~is_pred_matched)[0]
        if unmatched_pred_inds.shape[0] > 0:
            self.logger.debug(f"Prediction {unmatched_pred_inds} is not matched to any target.")
            pairing[-1] = unmatched_pred_inds.tolist()

        return pairing

    def valid_mask(
            self,
            exp_name: str,
            recon_curve_continuous_labels: np.ndarray[int],
            original_curve_continuous_labels: np.ndarray[int],
            extend_deny_regions: bool = True
    ) -> np.ndarray[bool]:
        """
        Create a boolean mask from deny regions.

        Args:
            exp_name: The name of the experiment session.
            recon_curve_continuous_labels: The reconstructed blink labels in continuous form.
            original_curve_continuous_labels: The original blink labels in continuous form.
            extend_deny_regions: Whether to extend deny regions to cover blinks that are partially outside the deny regions. Default is True.

        Returns:
            A boolean mask where True indicates valid data points.
        """
        mask = np.ones_like(recon_curve_continuous_labels, dtype=bool)

        # filter out deny regions that are outside the crop range, and truncate deny regions that are partially outside
        sed = single_dataset_factory(
            data_folder=self.dataset_folder,
            experiment_name=exp_name,
            config=self.validation_dataset_config,
        )
        cropped_deny_regions = []
        crop_start, _ = sed.crop_to_start_end_index(self.val_crop)
        for deny_start, deny_end in self.rf_deny_list.get(exp_name, []):
            cropped_deny_start = deny_start - crop_start
            cropped_deny_end = deny_end - crop_start

            if cropped_deny_start < 0 and cropped_deny_end < 0:
                continue

            cropped_deny_regions.append((
                max(0, cropped_deny_start),
                max(0, cropped_deny_end)
            ))

        # some deny regions do not fully cover blinks, so we need to extend the deny regions
        if extend_deny_regions:
            non_blinking_mask = np.logical_and(
                recon_curve_continuous_labels == blink_defs.NON_BLINKING,
                original_curve_continuous_labels == blink_defs.NON_BLINKING
            )

            extended_deny_regions = []
            for deny_start, deny_end in cropped_deny_regions:
                mask_left = deny_start
                mask_right = deny_end

                if not non_blinking_mask[deny_start]:
                    mask_left = np.where(non_blinking_mask[:deny_start])[0][-1]

                if not non_blinking_mask[deny_end]:
                    mask_right = np.where(non_blinking_mask[deny_end:])[0][0] + deny_end + 1

                extended_deny_regions.append((mask_left, mask_right))
        else:
            extended_deny_regions = cropped_deny_regions

        for start, end in extended_deny_regions:
            mask[start:end] = False

        return mask

    @staticmethod
    def get_correlation(
            cropped_mask: np.ndarray[bool],
            reconstructed_curve: np.ndarray[float],
            original_curve: np.ndarray[float],
    ) -> dict[str, float]:
        """
        Calculate the Pearson correlation between the reconstructed curve and the original curve.

        Args:
            cropped_mask: A boolean mask converted from the cropped deny regions (not extended).
            reconstructed_curve: The reconstructed openness curve.
            original_curve: The original openness curve.

        Returns:
            A dictionary containing the correlation and p-value.
        """
        assert reconstructed_curve.shape[0] == original_curve.shape[
            0], "Curve length mismatch. Recon {}, Ori {}".format(
            reconstructed_curve.shape[0], original_curve.shape[0]
        )

        res = pearsonr(reconstructed_curve[cropped_mask], original_curve[cropped_mask])
        return {
            "correlation": res.statistic,
            "p_value": res.pvalue,
        }

    @staticmethod
    def get_rmse_and_mae(
            cropped_mask: np.ndarray[bool],
            reconstructed_curve: np.ndarray[float],
            original_curve: np.ndarray[float],
    ) -> dict[str, float]:
        """
        Calculate the RMSE and MAE between the reconstructed curve and the original curve.

        Args:
            cropped_mask: A boolean mask converted from the cropped deny regions (not extended).
            reconstructed_curve: The reconstructed openness curve.
            original_curve: The original openness curve.

        Returns:
            A dictionary containing the RMSE and MAE.
        """
        assert reconstructed_curve.shape[0] == original_curve.shape[
            0], "Curve length mismatch. Recon {}, Ori {}".format(
            reconstructed_curve.shape[0], original_curve.shape[0]
        )

        rmse = np.sqrt(np.mean((reconstructed_curve[cropped_mask] - original_curve[cropped_mask]) ** 2))
        mae = np.mean(np.abs(reconstructed_curve[cropped_mask] - original_curve[cropped_mask]))
        return {
            "rmse": rmse,
            "mae": mae,
        }

    @staticmethod
    def get_event_detection_metrics(
            pairing: Pairing,
    ) -> dict[str, int]:
        """
        Blink monotonic event detection metrics.

        Args:
            pairing: The pairing of predicted and target blink segments.

        Returns:
            A dictionary containing true positives, false positives, false negatives, and true negatives.
        """
        result = {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 0,
        }

        for target_ind, pred_inds in pairing.items():
            if target_ind < 0:
                # dummy empty target non-blink for mismatched pred segments.
                # these are false positives.
                result["false_positive"] += len(pred_inds)
            else:
                for pred_ind in pred_inds:
                    if pred_ind < 0:
                        # dummy empty pred non-blink for mismatched target segments.
                        # these are false negatives.
                        result["false_negative"] += 1
                    else:
                        result["true_positive"] += 1
        return result

    @staticmethod
    def get_partial_event_detection_metrics(
            pairing: Pairing,
            partial_blink_threshold: float,
            pred_blink_segments: list[BlinkSegment],
            target_blink_segments: list[BlinkSegment],
            recon_curve: np.ndarray[float],
            original_curve: np.ndarray[float],
    ) -> dict[str, int]:
        result = {
            "partial_true_positive": 0,
            "partial_false_positive": 0,
            "partial_false_negative": 0,
            "partial_true_negative": 0,
        }

        for target_ind, pred_inds in pairing.items():
            if target_ind < 0:
                # dummy empty target non-blink for mismatched pred segments.
                # these are false positives.
                for pred_ind in pred_inds:
                    pred_blink_segment = pred_blink_segments[pred_ind]
                    recon_curve_segment = recon_curve[pred_blink_segment.start:pred_blink_segment.end]
                    is_partial = np.min(recon_curve_segment) < partial_blink_threshold

                    if is_partial:
                        result["partial_false_positive"] += 1
                    else:
                        result["partial_true_negative"] += 1
            else:
                target_blink_segment = target_blink_segments[target_ind]
                curve_segment = original_curve[target_blink_segment.start:target_blink_segment.end]
                is_target_partial = np.min(curve_segment) < partial_blink_threshold

                for pred_ind in pred_inds:
                    if pred_ind < 0:
                        # dummy empty pred non-blink for mismatched target segments.
                        # these are false negatives.
                        if is_target_partial:
                            result["partial_false_negative"] += 1
                        else:
                            result["partial_true_negative"] += 1
                    else:
                        pred_blink_segment = pred_blink_segments[pred_ind]
                        recon_curve_segment = recon_curve[pred_blink_segment.start:pred_blink_segment.end]
                        is_partial = np.min(recon_curve_segment) < partial_blink_threshold

                        if is_target_partial and is_partial:
                            result["partial_true_positive"] += 1
                        elif is_target_partial and not is_partial:
                            result["partial_false_negative"] += 1
                        elif not is_target_partial and is_partial:
                            result["partial_false_positive"] += 1
                        else:
                            result["partial_true_negative"] += 1

        return result

    def get_blink_abs_and_rel_errors(
            self,
            pairing: Pairing,
            pred_blink_segments: list[BlinkSegment],
            target_blink_segments: list[BlinkSegment],
    ) -> dict[str, list[float]]:
        """
        Detailed blink phase analysis. Absolute and relative errors are calculated for each phase in a blink.

        Args:
            pairing: The pairing of predicted and target blink segments.
            pred_blink_segments: A list of predicted blink segments.
            target_blink_segments: A list of target blink segments.

        Returns:
            A dictionary containing absolute and relative errors for each phase in a blink.
        """

        closing_abs_errors = []
        interphase_abs_errors = []
        reopening_abs_errors = []

        closing_rel_abs_errors = []
        interphase_rel_abs_errors = []
        reopening_rel_abs_errors = []

        for target_ind, pred_inds in pairing.items():
            if target_ind < 0:
                # dummy empty target blink for mismatched pred segments.
                # these are false positives.
                for pred_ind in pred_inds:
                    pred_blink_segment = pred_blink_segments[pred_ind]
                    closing_abs_errors.append(pred_blink_segment.closing_duration)
                    interphase_abs_errors.append(pred_blink_segment.interphase_duration)
                    reopening_abs_errors.append(pred_blink_segment.reopening_duration)
            else:
                target_blink_segment = target_blink_segments[target_ind]

                if len(pred_inds) > 1:
                    self.logger.info(f"Multiple predictions matched to target {target_ind}: {pred_inds}")
                else:
                    pred_ind = pred_inds[0]
                    if pred_ind < 0:
                        # dummy empty pred blink for mismatched target segments.
                        # these are false negatives.
                        closing_abs_errors.append(target_blink_segment.closing_duration)
                        interphase_abs_errors.append(target_blink_segment.interphase_duration)
                        reopening_abs_errors.append(target_blink_segment.reopening_duration)
                    else:
                        pred_blink_segment = pred_blink_segments[pred_ind]

                        closing_abs_error = abs(
                            pred_blink_segment.closing_duration - target_blink_segment.closing_duration)
                        interphase_abs_error = abs(
                            pred_blink_segment.interphase_duration - target_blink_segment.interphase_duration)
                        reopening_abs_error = abs(
                            pred_blink_segment.reopening_duration - target_blink_segment.reopening_duration)

                        closing_abs_errors.append(closing_abs_error)
                        interphase_abs_errors.append(interphase_abs_error)
                        reopening_abs_errors.append(reopening_abs_error)

                        closing_rel_abs_errors.append(closing_abs_error / target_blink_segment.closing_duration)
                        interphase_rel_abs_errors.append(
                            interphase_abs_error / target_blink_segment.interphase_duration)
                        reopening_rel_abs_errors.append(reopening_abs_error / target_blink_segment.reopening_duration)

        return {
            "closing": closing_abs_errors,
            "interphase": interphase_abs_errors,
            "reopening": reopening_abs_errors,
            "closing_rel": closing_rel_abs_errors,
            "interphase_rel": interphase_rel_abs_errors,
            "reopening_rel": reopening_rel_abs_errors,
        }

    @staticmethod
    def get_non_blink_abs_and_rel_errors(
            pairing: Pairing,
            pred_non_blink_segments: list[NonBlinkSegment],
            target_non_blink_segments: list[NonBlinkSegment],
    ) -> dict[str, list[float]]:
        """
        Detailed blink phase analysis. Absolute and relative errors are calculated for open phases.

        Args:
            pairing: The pairing of predicted and target non-blink segments.
            pred_non_blink_segments: A list of predicted non-blink segments.
            target_non_blink_segments: A list of target non-blink segments.

        Returns:
            A dictionary containing absolute and relative errors for open phases.
        """
        non_blink_abs_error = []
        non_blink_rel_abs_error = []

        for target_ind, pred_inds in pairing.items():
            if target_ind < 0:
                # dummy empty target non-blink for mismatched pred segments.
                # these are false positives.
                for pred_ind in pred_inds:
                    non_blink_abs_error.append(pred_non_blink_segments[pred_ind].non_blinking_duration)
            else:
                target_non_blink_segment = target_non_blink_segments[target_ind]

                for pred_ind in pred_inds:
                    if pred_ind < 0:
                        # dummy empty pred non-blink for mismatched target segments.
                        # these are false negatives.
                        non_blink_abs_error.append(target_non_blink_segment.non_blinking_duration)
                    else:
                        non_blink_abs_error.append(
                            abs(
                                pred_non_blink_segments[pred_ind].non_blinking_duration
                                - target_non_blink_segment.non_blinking_duration
                            )
                        )
                        non_blink_rel_abs_error.append(
                            abs(
                                pred_non_blink_segments[pred_ind].non_blinking_duration
                                - target_non_blink_segment.non_blinking_duration
                            )
                            / target_non_blink_segment.non_blinking_duration
                        )

        return {
            "non_blinking": non_blink_abs_error,
            "non_blinking_rel": non_blink_rel_abs_error,
        }

    @staticmethod
    def get_iou(pred_labels_continuous, target_labels_continuous, num_classes=4):
        """
        Detailed blink phase analysis. Absolute and relative errors are calculated for each phase in a blink.

        Args:
            pred_labels_continuous: A numpy array of predicted labels in the continuous form.
            target_labels_continuous: A numpy array of target labels in the continuous form.
            num_classes: The number of blink phases. Should be 4.

        Returns:
            A numpy array containing IoU for each class.
        """
        # Get the confusion matrix
        cm = confusion_matrix(target_labels_continuous, pred_labels_continuous, labels=np.arange(num_classes))

        # Calculate IoU for each class
        ious = []
        for i in range(num_classes):
            intersection = cm[i, i]  # Diagonal element is the true positive for the class
            union = np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i]  # Union is TP + FP + FN
            iou = intersection / union if union != 0 else 1.0
            ious.append(iou)

        return {
            "closing_iou": ious[1],
            "interphase_iou": ious[2],
            "reopening_iou": ious[3],
            "non_blinking_iou": ious[0],
        }

    @validate_literal_args
    def process(
            self,
            exp_name: str,
            reconstructed_curve: np.ndarray[float],
            original_curve: np.ndarray[float],
            reconstructed_curve_labels: np.ndarray[int],
            original_curve_labels: np.ndarray[int],
            metrics: Sequence[Literal["correlation", "rmse_mae", "event", "phase_analysis", "partial_blinks"]] = (
            "correlation",),
            partial_blink_thresholds: Optional[Sequence[float]] = (0.1, 0.5, 0.9),
    ):
        """
        Process the metrics for the reconstructed curve.

        Args:
            exp_name: The name of the experiment to compute the metrics for.
            reconstructed_curve: The reconstructed openness curve.
            original_curve: The original openness curve.
            reconstructed_curve_labels: The reconstructed blink labels of phases.
            original_curve_labels: The original blink labels of phases.
            metrics: A sequence of metrics to compute. Options are "correlation", "rmse_mae", "event", "phase_analysis", and "partial_blinks".
            partial_blink_thresholds: A sequence of thresholds to define partial blinks. Default is (0.1, 0.5, 0.9). If "partial_blinks" is in metrics, this is required.

        Returns:
            A dictionary containing the computed metrics.
        """
        recon_curve_continuous_labels = from_spiky_labels_to_continuous_labels(reconstructed_curve_labels)
        original_curve_continuous_labels = from_spiky_labels_to_continuous_labels(original_curve_labels)

        metrics = list(set(metrics))

        total_results = {}

        valid_mask = self.valid_mask(
            exp_name,
            recon_curve_continuous_labels,
            original_curve_continuous_labels,
            extend_deny_regions=True
        )

        if "correlation" in metrics:
            total_results.update(self.get_correlation(valid_mask, reconstructed_curve, original_curve))

        if "rmse_mae" in metrics:
            total_results.update(self.get_rmse_and_mae(valid_mask, reconstructed_curve, original_curve))

        if "event" or "phase_analysis" in metrics:
            reconstructed_curve_labels = reconstructed_curve_labels[valid_mask]
            original_curve_labels = original_curve_labels[valid_mask]

            blink_segments = self.labels_to_blink_segments(reconstructed_curve_labels)
            original_blink_segments = self.labels_to_blink_segments(original_curve_labels)
            blink_pairing = self.pair_segments(blink_segments, original_blink_segments)

            if "event" in metrics:
                total_results.update(self.get_event_detection_metrics(blink_pairing))

            if "phase_analysis" in metrics:
                non_blink_segments = self.labels_to_non_blink_segments(reconstructed_curve_labels)
                original_non_blink_segments = self.labels_to_non_blink_segments(original_curve_labels)
                non_blink_pairing = self.pair_segments(non_blink_segments, original_non_blink_segments)

                total_results.update(
                    self.get_blink_abs_and_rel_errors(blink_pairing, blink_segments, original_blink_segments))
                total_results.update(self.get_non_blink_abs_and_rel_errors(non_blink_pairing, non_blink_segments,
                                                                           original_non_blink_segments))
                total_results.update(self.get_iou(recon_curve_continuous_labels, original_curve_continuous_labels))

            if "partial_blinks" in metrics:
                assert partial_blink_thresholds is not None, "Partial blink thresholds must be provided for partial_blinks metric."

                partial_blink_thresholds = np.array(partial_blink_thresholds)
                assert np.all(np.logical_and(partial_blink_thresholds >= 0., partial_blink_thresholds <= 1.)), "The values in the list must be between 0 and 1"

                partial_results = []
                for threshold in partial_blink_thresholds:
                    partial_results_at_a_threshold = self.get_partial_event_detection_metrics(
                        pairing=blink_pairing,
                        partial_blink_threshold=threshold,
                        pred_blink_segments=blink_segments,
                        target_blink_segments=original_blink_segments,
                        recon_curve=reconstructed_curve,
                        original_curve=original_curve,
                    )
                    partial_results.append({
                        "experiment": exp_name,
                        "threshold": threshold,
                        "partial_true_positive": partial_results_at_a_threshold["partial_true_positive"],
                        "partial_false_positive": partial_results_at_a_threshold["partial_false_positive"],
                        "partial_false_negative": partial_results_at_a_threshold["partial_false_negative"],
                        "partial_true_negative": partial_results_at_a_threshold["partial_true_negative"],
                    })

                total_results["partial_blinks"] = partial_results

        return total_results
