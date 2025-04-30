import json
import logging
import math
import re
from pathlib import Path
from typing import NamedTuple, Sequence, Union

import numpy as np
import pandas as pd
from intervaltree import IntervalTree, Interval
from prettytable import PrettyTable
from scipy import interpolate
from tqdm import tqdm

from src.core import constants, project_files, setup_logging, blink_defs
from .dataset_specification import ProcessingConfig, ProcessingProtocol, RadarDataProcessor, VisionDataProcessor
from ..event_proposal.event_proposer import EventProposer


# =========================
# helper functions
# =========================

def list_experiment_folders(data_folder: Union[str, Path], exp_folder_name_pattern: list[str]) -> list[str]:
    """
    List experiment folders in the data folder that match the given pattern.

    Args:
        data_folder: Path to the root folder containing the experiment data.
        exp_folder_name_pattern: List of strings to match the experiment folder names.

    Returns:
        List of experiment folder names that match the given pattern.
    """
    # Combine the patterns into a single regex
    combined_pattern = '|'.join(map(re.escape, exp_folder_name_pattern))
    combined_regex = re.compile(combined_pattern)

    # Use Path and filter directories matching the regex
    data_path = Path(data_folder)
    experiment_folder_names = sorted([
        folder.name
        for folder in data_path.iterdir()
        if folder.is_dir() and re.search(combined_regex, folder.name)
    ])
    return experiment_folder_names


def crop_valid_session(
        first_event_ts_on_laptop: float,
        last_event_ts_on_laptop: float,
        vframe_label: np.ndarray[int],
        vt: np.ndarray[float]
) -> tuple[int, int]:
    """
    Crop out the valid session from the video labels.

    If event timestamps are inside a blink, they will be adjusted to the nearest non-blinking frame.

    Args:
        first_event_ts_on_laptop: Timestamp that marks the start of the first event.
        last_event_ts_on_laptop: Timestamp that marks the end of the last event.
        vframe_label: Video frame labels.
        vt: Video timestamps.

    Returns:
        Tuple of the start and end indices of the valid session.
    """
    n_vframes = vt.shape[0]
    first_event_index_on_vframe = max(0, math.ceil(
        (first_event_ts_on_laptop - vt[0]) * constants.DEFAULT_CORRECTED_VIDEO_FPS))
    last_event_index_on_vframe = min(n_vframes, math.floor(
        (last_event_ts_on_laptop - vt[0]) * constants.DEFAULT_CORRECTED_VIDEO_FPS))

    # if there is any CLOSING labeling before the first event, use the last one as the real start index
    # otherwise, use the first CLOSING labeling after the first event as the real start index
    before_first_event = vframe_label[:first_event_index_on_vframe]
    before_first_event_closing_labels = np.where(before_first_event == blink_defs.CLOSING)[0]
    if before_first_event_closing_labels.shape[0] > 0:
        # offset a bit to include some non-blinking data
        actual_label_start_index = max(0, before_first_event_closing_labels[-1] - 50)
    else:
        after_first_event = vframe_label[max(0, first_event_index_on_vframe):]
        after_first_event_closing_labels = np.where(after_first_event == blink_defs.CLOSING)[
                                               0] + first_event_index_on_vframe
        # offset a bit to include some non-blinking data
        actual_label_start_index = max(0, after_first_event_closing_labels[0] - 50)

    # if there is any NON_BLINKING labeling after the last event, use the first one as the real end index
    # otherwise, use the last NON_BLINKING labeling before the last event as the real end index
    after_last_event = vframe_label[last_event_index_on_vframe:]
    after_last_event_non_blinking_labels = np.where(
        after_last_event == blink_defs.NON_BLINKING
    )[0] + last_event_index_on_vframe
    if after_last_event_non_blinking_labels.shape[0] > 0:
        # offset a bit to include some non-blinking data
        actual_label_end_index = min(n_vframes, after_last_event_non_blinking_labels[0] + 50)
    else:
        before_last_event = vframe_label[:last_event_index_on_vframe]
        before_last_event_non_blinking_labels = np.where(before_last_event == blink_defs.NON_BLINKING)[0]
        # offset a bit to include some non-blinking data
        actual_label_end_index = min(n_vframes, before_last_event_non_blinking_labels[-1] + 50)
    return actual_label_end_index, actual_label_start_index


def make_cluster(labels: np.ndarray[int]) -> list[tuple[int, int]]:
    """
    Create clusters of positive frames (blinks) from a label array.

    Args:
        labels: An array of labels that follows the ``blink_defs``.

    Returns:
        A list of tuples where each tuple represents the start and end indices of a cluster, inclusive.
    """
    if np.issubdtype(labels.dtype, bool):
        labels - labels.astype(int)

    blink_frame_indices = np.where(labels > 0)[0]
    cluster_starts = blink_frame_indices[np.where(np.diff(blink_frame_indices, prepend=-2) > 1)[0]]
    cluster_ends = blink_frame_indices[
        np.where(np.diff(blink_frame_indices, append=labels.shape[0] + 2) > 1)[0]]

    assert cluster_starts.shape[0] == cluster_ends.shape[0], "Unmatched #starts and #ends for clusters."

    return list(zip(cluster_starts, cluster_ends))


def build_interval_tree(cluster_list: Sequence[tuple[int, int]]) -> IntervalTree:
    """
    Build an interval tree from a list of clusters.

    Args:
        cluster_list: A list of tuples where each tuple represents the start and end indices of a cluster, inclusive.

    Returns:
        An interval tree built from the cluster list.
    """
    return IntervalTree(Interval(begin, end + 1) for begin, end in cluster_list)


def merge_event_detector_with_gt(predicted: np.ndarray[int], ground_truth: np.ndarray[int]) -> np.ndarray[int]:
    """
    Merge the predicted events with the ground truth events.

    Predicted events overlapping ground truth are marked 1; non-overlapping are marked ``-2``.
    ``-2`` are hard examples representing non-blinking movements, like eyeball movements or face structure changes.

    Args:
        predicted: Predicted events from the event proposal algorithm. Binary labels.
        ground_truth: Manual labels from the video data. Follows the ``blink_defs``.

    Returns:
        An array of merged labels.
    """
    # Create clusters and interval trees
    predicted_clusters = make_cluster(predicted)
    ground_truth_clusters = make_cluster(ground_truth)
    truth_tree = build_interval_tree(ground_truth_clusters)

    # Initialize result array with the original predictions
    result = predicted.copy()

    # Validate predicted events
    for start, end in predicted_clusters:
        # Check if the predicted event overlaps with any ground truth event
        if not truth_tree.overlaps(start, end + 1):
            # If no overlap, mark as false positive (-2)
            result[start:end + 1] = -2

    return result


# =========================
# main function
# =========================

def construct_dataset(
        data_folder: Union[str, Path],
        output_folder: Union[str, Path],
        experiment_dates: list[str] = None,
        processing_protocol: ProcessingProtocol = ("range-querying-fft", "low-pass-filtering", "diff", "normalization"),
        processing_config: Union[str, dict, ProcessingConfig] = None,
        append_protocol_suffix_to_output_folder: bool = True,
        verbose: bool = False,
):
    """
    Construct dataset for the openness prediction model.

    Args:
        data_folder: Path to the root folder containing the experiment data.
        output_folder: Path to the output folder where the dataset will be saved.
        experiment_dates: List of experiment dates to process. If None, all experiments in the data folder will be processed.
        processing_protocol: Processing protocol to apply to the radar data. Default is ("range-querying-fft", "low-pass-filtering", "diff", "normalization").
        processing_config: Processing configuration to use for the radar data. Can be a string (path to a JSON file), a dictionary, or a ProcessingConfig object.
        append_protocol_suffix_to_output_folder: Whether to append the processing protocol suffix to the output folder name. Default is True.
        verbose: Whether to print logs. Default is False.
    """
    # =========================
    # initialization
    # =========================
    # set up logging
    setup_logging(None, level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger(__name__)

    data_folder = Path(data_folder)
    output_folder = Path(output_folder)

    # initialize data processors
    radar_data_processor = RadarDataProcessor(
        processing_protocol=processing_protocol,
        processing_config=processing_config
    )
    vision_data_processor = VisionDataProcessor()

    logger.info("Processing protocol: {}".format(" -> ".join(processing_protocol)))
    logger.info("Processing configuration:")
    for k, v in radar_data_processor.processing_config.__dict__.items():
        logger.info(f"{k} -> {v}")

    # list experiment folders to be processed
    if experiment_dates is None:
        exp_folders = [folder.name for folder in data_folder.iterdir() if
                       folder.is_dir() and folder.name.startswith("exp")]
        if len(exp_folders) == 0:
            logger.warning("No experiment data in this folder. Please check the input root path.")
            return
    else:
        exp_folders = list_experiment_folders(data_folder, experiment_dates)
    exp_folders = list(sorted(exp_folders))

    # load deny lists.
    deny_list_path = data_folder / project_files.deny_list_filename
    deny_list = {}
    rf_deny_list = {}

    if deny_list_path.exists():
        with deny_list_path.open() as f:
            deny_list = json.load(f)
    else:
        logger.warning(f"No deny list file found @ {deny_list_path}. All captured data will be used.")

    # manage output path, and global files to output directly in the folder.
    if append_protocol_suffix_to_output_folder:
        suffix_parts = []
        for protocol in processing_protocol:
            # Split the protocol into words and get the first letter of each word
            parts = protocol.split('-')
            first_letters = ''.join(word[0] for word in parts)
            suffix_parts.append(first_letters)
        # Join all parts with hyphen to form the suffix
        suffix = '_'.join(suffix_parts)
        # Append the suffix to the output_folder
        output_folder = output_folder.with_name(output_folder.name + f"_{suffix}")
    #   create output folder
    if not output_folder.exists():
        logging.info(f"Output folder {output_folder} does not exist. Creating now...")
        output_folder.mkdir(parents=True)

    #   deny list in RF frame timestamps
    rf_deny_list_path = output_folder / project_files.rf_deny_list_filename
    processing_config_path = output_folder / project_files.dataset_processing_config_filename

    # =========================
    # processing each individual experiment
    # =========================
    FailedExperiment = NamedTuple("FailedExperiment", [("exp_folder", str), ("reason", str)])
    failed_experiments: list[FailedExperiment] = []
    for exp_folder in tqdm(exp_folders):
        # -------------------------
        # prepare all paths
        # -------------------------
        data_exp_folder = data_folder / exp_folder
        output_exp_folder = output_folder / exp_folder
        if not output_exp_folder.exists():
            output_exp_folder.mkdir()

        model_input_path = output_exp_folder / project_files.model_input_filename
        model_label_path = output_exp_folder / project_files.model_label_filename
        model_target_ratio_path = output_exp_folder / project_files.model_target_ratio_filename
        protocol_path = output_exp_folder / project_files.dataset_processing_protocol_filename
        # -------------------------
        # load experiment data
        # -------------------------
        #       video data
        try:
            vt = np.load(data_exp_folder / project_files.video_timestamps_filename)
        except FileNotFoundError:
            failed_experiments.append(FailedExperiment(exp_folder, "Video timestamps not found"))
            continue
        n_vframes = vt.shape[0]
        label_df = pd.read_csv(data_exp_folder / project_files.video_label_dataframe_filename, index_col=[0])

        try:
            vframe_label = label_df["init_labels"].to_numpy()
        except KeyError:
            failed_experiments.append(FailedExperiment(exp_folder, "Label column (init_labels) not found in dataframe"))
            continue

        #       radar data
        rd = np.load(data_exp_folder / project_files.radar_raw_data_filename)
        rt = np.load(data_exp_folder / project_files.radar_timestamps_filename)

        #       special timing in the experiment
        try:
            with (data_exp_folder / project_files.metadata).open() as f:
                metadata = json.load(f)
        except FileNotFoundError:
            failed_experiments.append(FailedExperiment(exp_folder, "Metadata file not found"))
            continue

        first_event_ts_on_laptop = metadata["laptop_entire_start"][0] - metadata["laptop_timestamp_offset"]
        last_event_ts_on_laptop = metadata["laptop_entire_end"][0] - metadata["laptop_timestamp_offset"]

        # -------------------------
        # crop out data within the experiment data window
        # -------------------------
        actual_label_end_index, actual_label_start_index = crop_valid_session(
            first_event_ts_on_laptop, last_event_ts_on_laptop, vframe_label, vt
        )
        valid_vt = vt[actual_label_start_index:actual_label_end_index]
        valid_vframe_labels = vframe_label[actual_label_start_index:actual_label_end_index]

        # if vision score is checked to save, prepare here
        valid_vision_score_ratio_dict = {}
        for score_name in ("projected_blink_ratio",):
            vision_score = label_df[f"left_{score_name}"].to_numpy()
            valid_vision_score = vision_score[actual_label_start_index:actual_label_end_index]

            #      the original labels only marks the beginning of each phase.
            #      we fill between phase starts with the same label.
            valid_vframe_labels_full = np.full_like(valid_vframe_labels, blink_defs.NON_BLINKING)
            all_phase_start_indices = np.where(valid_vframe_labels != blink_defs.NOT_INITIALIZED)[0]
            for i in range(all_phase_start_indices.shape[0] - 1):
                valid_vframe_labels_full[all_phase_start_indices[i]:all_phase_start_indices[i + 1]] = \
                    valid_vframe_labels[all_phase_start_indices[i]]
            valid_vframe_labels_full[all_phase_start_indices[-1]:] = valid_vframe_labels[all_phase_start_indices[-1]]

            #       process.
            valid_vision_score_ratio = vision_data_processor.normalize_vision_data(
                valid_vision_score, valid_vframe_labels_full
            )
            valid_vision_score_ratio_dict[score_name] = valid_vision_score_ratio

        # -------------------------
        # create labels for radar data
        # -------------------------
        # check whether the radar timestamp is monotonically increasing
        decrease_mask = np.diff(rt) < 0
        decrease_index = np.argmax(decrease_mask) + 1 if np.any(decrease_mask) else len(rt)
        monotonic_mask = np.arange(len(rt)) < decrease_index

        valid_rt_mask = (rt >= valid_vt[0]) & (rt < valid_vt[-1])
        valid_rt_mask = valid_rt_mask & monotonic_mask
        if np.any(decrease_mask):
            failed_experiments.append(
                FailedExperiment(exp_folder, f"Non-monotonic radar timestamp detected @ {decrease_index} / {len(rt)}"))
            continue

        valid_rt = rt[valid_rt_mask]
        valid_rd_labels = np.zeros((valid_rt.shape[0],), dtype=np.int32)

        # we have included additional data. these are added to make sure all data are covered with labels
        if valid_vframe_labels[0] == -1:
            valid_vframe_labels[0] = blink_defs.NON_BLINKING
        if valid_vframe_labels[-1] == -1:
            valid_vframe_labels[-1] = blink_defs.NON_BLINKING

        # mark labels
        all_phase_start_indices = np.where(valid_vframe_labels != -1)[0]
        for i in range(all_phase_start_indices.shape[0] - 1):
            phase_start_time = valid_vt[all_phase_start_indices[i]]
            phase_end_time = valid_vt[all_phase_start_indices[i + 1]]
            phase_rd_mask = (valid_rt >= phase_start_time) & (valid_rt < phase_end_time)
            valid_rd_labels[phase_rd_mask] = valid_vframe_labels[all_phase_start_indices[i]]

        # interpolate vision score to match dimension of radar signals
        valid_rd_target_ratio_dict = {}
        for score_name in ("projected_blink_ratio",):
            valid_vision_score_ratio = valid_vision_score_ratio_dict[score_name]
            interp_func_ratio = interpolate.interp1d(valid_vt, valid_vision_score_ratio, axis=0, kind='cubic')
            valid_rd_target_ratio = interp_func_ratio(valid_rt)
            valid_rd_target_ratio_dict[score_name] = valid_rd_target_ratio

        # -------------------------
        # processing of radar data
        # -------------------------
        #   full label is needed for projected-diff.
        processed_rd = radar_data_processor.apply(rd)

        # add additional information to labels.
        # they mark hard examples, i.e., where non-blink eye movements happen.
        if not any(map(lambda c: "norm" in c, processing_protocol)):
            # if normalization is included, directly the threshold can be used.
            energy = np.mean(np.abs(processed_rd), axis=tuple(range(1, processed_rd.ndim)))
        else:
            # if norm is not included, use normed data for detection.
            normed = radar_data_processor._normalize(processed_rd)
            energy = np.mean(np.abs(normed), axis=tuple(range(1, processed_rd.ndim)))

        is_event_mask = energy >= 0.03
        event_detector = EventProposer(
            batch_shape=(1,),
            idle_buffer_n_batches=32,
            i2a_n_pos=45,
            i2a_n_neg=2,
            a2i_n_pos=64,
            a2i_n_neg=64
        )
        fsm_labels = event_detector.detect(is_event_mask)

        # additional processing.
        # mark overlap fsm-only detected events to -2, and mark overlap with real blinks as 1.
        merged_fsm_labels = merge_event_detector_with_gt(fsm_labels[valid_rt_mask], valid_rd_labels).astype(np.int32)

        # -------------------------
        # some more processing about the valid radar data.
        # -------------------------
        #       crop.
        valid_rd = processed_rd[valid_rt_mask]
        #       reshape. channels (antennas) last.
        valid_rd = np.squeeze(valid_rd)
        n_dims = len(valid_rd.shape)
        valid_rd = valid_rd.transpose((0,) + tuple(range(2, n_dims)) + (1,))

        # -------------------------
        # materialize inputs and labels
        # -------------------------
        with protocol_path.open('w') as f:
            for protocol_component in processing_protocol:
                f.write(protocol_component + '\n')

        np.save(model_input_path, valid_rd)
        np.savez_compressed(model_label_path, video=valid_rd_labels, fsm=merged_fsm_labels)
        np.savez_compressed(model_target_ratio_path, **valid_rd_target_ratio_dict)

        # -------------------------
        # copy experiment metadata
        # -------------------------
        with (output_exp_folder / project_files.metadata).open('w') as f:
            json.dump(metadata, f, indent=4)

        # -------------------------
        # convert deny list from video frame number to rf frame number
        # -------------------------
        # source 1: manually labelled intervals where data should be excluded,
        exp_deny_list = deny_list.get(exp_folder, [])

        # conversion to radar data timestamps
        if len(exp_deny_list) > 0:
            converted_intervals = []
            for denied_start, denied_end in exp_deny_list:
                vt_denied_start, vt_denied_end = vt[denied_start], vt[denied_end]

                rt_before_denied_start = valid_rt[valid_rt <= vt_denied_start]
                if rt_before_denied_start.size == 0:
                    continue
                rt_denied_start = int(np.where(valid_rt == np.max(rt_before_denied_start))[0][0])

                rt_after_denied_end = valid_rt[valid_rt >= vt_denied_end]
                if rt_after_denied_end.size == 0:
                    continue
                rf_denied_end = int(np.where(valid_rt == np.min(rt_after_denied_end))[0][0])

                converted_intervals.append((rt_denied_start, rf_denied_end))
            rf_deny_list[exp_folder] = converted_intervals

    table = PrettyTable()
    table.field_names = ["Experiment", "Status"]
    for failed_exp in failed_experiments:
        table.add_row([failed_exp.exp_folder, failed_exp.reason])
    logger.info(f"Failed experiments:")
    logger.info("\n" + table.get_string(sortby="Experiment"))

    # =========================
    # materialize dataset level files
    # =========================
    #  deny list in RF frame timestamps
    logger.info("Full deny list in RF frame unit:")
    for k, v in rf_deny_list.items():
        logger.info(f"\t {k} -> {v}")
    with rf_deny_list_path.open("w") as f:
        json.dump(rf_deny_list, f, indent=4)

    #  processing configuration used
    radar_data_processor.processing_config.save_to_json(processing_config_path)

    logger.info("Dataset formatting done.")
