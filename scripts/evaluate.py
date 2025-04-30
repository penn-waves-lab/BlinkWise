import argparse
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

script_dir = Path(__file__).resolve().parent
project_folder = script_dir.parent
if str(project_folder) not in sys.path:
    sys.path.insert(0, str(project_folder))

from src.core import project_files, blink_defs, constants
from src.models.data import single_dataset_factory
from src.models.config import ExperimentConfig
from src.models.utils import prepare_exp_subsets, summarize_subjects_and_experiments
from src.evaluation import curve_reconstructor_factory, labeler_factory, MetricEvaluator


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate configuration")
    parser.add_argument(
        "-r", "--result-folder", help="Path to the result folder", required=True,
    )
    parser.add_argument(
        "-o", "--output-folder",
        help="Path to the output folder. If not provided, the output will be saved in the same folder as the result folder",
    )
    parser.add_argument(
        "-d", "--dataset-folder",
        help="Path to the dataset folder. If not provided, the dataset folder will be loaded from the experiment configuration file",
    )
    parser.add_argument(
        "-t", "--tflite", help="Whether the result file is in TFLite format", action="store_true",
    )
    parser.add_argument(
        "-v", "--verbose", help="Whether to print logs", action="store_true",
    )
    parser.add_argument(
        "--overwrite-curves", help="Overwrite the predicted curves if they already exist", action="store_true",
    )
    parser.add_argument(
        "--overwrite-labels", help="Overwrite the labels if they already exist", action="store_true",
    )
    parser.add_argument(
        "--overwrite-metrics", help="Overwrite the metrics if they already exist", action="store_true",
    )

    return parser.parse_args()


def load_checkpoint_path(result_folder: Path, tflite: bool) -> Path:
    """
    Load the appropriate model checkpoint path based on the model type.
    
    Args:
        result_folder: Path to the folder containing the model results
        tflite: Whether to load a TFLite model or not
    
    Returns:
        Path to the checkpoint file
    
    Raises:
        FileNotFoundError: If the requested checkpoint cannot be found
    """
    if tflite:
        # TFLite model requested
        default_path = result_folder / "checkpoint" / "quantized_model.tflite"
        if default_path.exists():
            return default_path
        
        # Check QAT folder for quantized models
        qat_folder = result_folder / "qat"
        if not qat_folder.exists():
            raise FileNotFoundError(f"TFLite model requested but QAT folder not found at: {qat_folder}. "
                                    f"Please run the quantization process first.")
        
        qat_experiment_folders = [f for f in qat_folder.iterdir() if f.is_dir()]
        if not qat_experiment_folders:
            raise FileNotFoundError(f"TFLite model requested but no QAT sessions found in: {qat_folder}. "
                                    f"Please run the quantization process first.")
        
        checkpoint_path = qat_experiment_folders[0] / "quantized_model.tflite"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"TFLite model not found at expected location: {checkpoint_path}. "
                                    f"The QAT process may not have completed successfully.")
        
        return checkpoint_path
    else:
        # Keras model requested
        checkpoint_path = result_folder / "checkpoint" / "unet.keras"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Keras model not found at: {checkpoint_path}. "
                                    f"Please ensure the model was trained and saved correctly.")
        
        return checkpoint_path


def load_original_curve_and_non_blinking_masks(
        dataset_folder: Path,
        exp_name: str,
        dataset_config,
        crop: tuple[float, float]
) -> tuple[np.ndarray[float], np.ndarray[bool]]:
    """
    Load the original curve and non-blinking masks for a given experiment.

    Args:
        dataset_folder: Path to the processed dataset folder
        exp_name: Name of the experiment
        dataset_config: Configuration object for the dataset
        crop: Tuple of start and end indices for the crop range

    Returns:

    """
    sed = single_dataset_factory(data_folder=dataset_folder, experiment_name=exp_name, config=dataset_config)
    original_curve = sed._get_target_data(crop, dataset_config.curve_processing_protocol)

    crop_start, crop_end = sed.crop_to_start_end_index(crop)
    label_file_path = dataset_folder / exp_name / project_files.model_label_filename

    with np.load(label_file_path) as label_data:
        video = label_data["video"][crop_start:crop_end]

    video_non_blinking_mask = video == blink_defs.NON_BLINKING

    return original_curve, video_non_blinking_mask


def predict_curve(
        subject_exp_dict: dict[str, list[str]],
        exp_config_path: Path,
        checkpoint_path: Path,
        dataset_folder: Path,
        output_folder: Path = None,
        verbose: bool = False,
        overwrite: bool = False,
) -> Path:
    """
    Step 1: Predict openness curves for the given subjects and experiments from RF dataset.

    Args:
        subject_exp_dict: Dictionary of subjects and their corresponding experiments.
        exp_config_path: Path to the experiment configuration file.
        checkpoint_path: Path to the model checkpoint file.
        dataset_folder: Path to the processed dataset folder.
        output_folder: Path to the output folder of predicted openness curves. Curves are gathered in `curve_reconstructor_[keras/tflite]` depending on which checkpoint is used.
        verbose: Whether to print logs.
        overwrite: Whether to overwrite the predicted openness curves if they already exist.

    Returns:
        Path to the output folder of predicted openness curves.
    """
    curve_reconstructor = curve_reconstructor_factory(
        exp_config_path=exp_config_path,
        checkpoint_path=checkpoint_path,
        dataset_folder=dataset_folder,
        output_folder=output_folder,
        verbose=verbose,
    )

    for subject, exp_list in tqdm(subject_exp_dict.items(), desc="Predicting curves"):
        for exp in exp_list:
            # loop through each subject and experiment

            # check if the openness curve already exists
            output_curve_file = curve_reconstructor.output_folder / f"{subject}_{exp}.npy"
            if output_curve_file.exists() and not overwrite:
                curve_reconstructor.logger.info(f"Curve already exists for {subject}_{exp}. Skipping...")
                continue

            if not curve_reconstructor.output_folder.exists():
                curve_reconstructor.output_folder.mkdir(parents=True)

            # predict the openness curve
            curve = curve_reconstructor.process(
                exp_name=exp,
                crop_type="validation",
                post_processing=True,
            )

            # save the openness curve
            np.save(output_curve_file, curve)

    return curve_reconstructor.output_folder


def label_curve(
        subject_exp_dict: dict[str, list[str]],
        exp_config_path: Path,
        dataset_folder: Path,
        curve_output_folder: Path,
        output_folder: Path = None,
        verbose: bool = False,
        is_tflite: bool = False,
        overwrite: bool = False,
) -> Path:
    """
    Step 2: Label the predicted openness curves with blink phase information for the given subjects and experiments.

    Args:
        subject_exp_dict: Dictionary of subjects and their corresponding experiments.
        exp_config_path: Path to the experiment configuration file.
        dataset_folder: Path to the processed dataset folder.
        curve_output_folder: Path to the output folder of predicted openness curves. Labels are saved under `labeling` folder as npz files. TF Lite labels have `_tflite` suffix.
        output_folder: Path to the output folder of blink phase labels.
        verbose: Whether to print logs.
        is_tflite: Whether to process curves predicted by TFLite model or not.
        overwrite: Whether to overwrite the labels if they already exist.

    Returns:
        Path to the output folder of blink phase labels.
    """
    gt_labeler = labeler_factory(
        labeler_name="flank",
        exp_config_path=exp_config_path,
        peak_quantile=0.8,
        fps=int(constants.RADAR_FPS),
        min_peak_distance=(100 / int(constants.RADAR_FPS)),
        output_folder=output_folder,
        verbose=verbose,
    )
    pred_labeler = labeler_factory(
        labeler_name="flank",
        exp_config_path=exp_config_path,
        fps=int(constants.RADAR_FPS),
        min_peak_distance=(100 / int(constants.RADAR_FPS)),
        output_folder=output_folder,
        verbose=verbose,
    )

    for subject, exp_list in tqdm(subject_exp_dict.items(), desc="Labeling curves"):
        for exp in exp_list:
            # loop through each subject and experiment
            output_curve_file = curve_output_folder / f"{subject}_{exp}.npy"
            if not output_curve_file.exists():
                gt_labeler.logger.info(f"Curve does not exist for {subject}_{exp}. Skipping...")
                continue

            if not gt_labeler.output_folder.exists():
                gt_labeler.output_folder.mkdir(parents=True)

            # load the predicted curve and original curve
            pred_curve = np.load(output_curve_file)
            original_curve, mask = load_original_curve_and_non_blinking_masks(
                dataset_folder=dataset_folder,
                exp_name=exp,
                dataset_config=gt_labeler.validation_dataset_config,
                crop=gt_labeler.training_config.get_crop_range("validation"),
            )

            # check if the labels already exist
            label_output_file = gt_labeler.output_folder / f"{subject}_{exp}{'_tflite' if is_tflite else ''}.npz"

            if label_output_file.exists() and not overwrite:
                gt_labeler.logger.info(f"Labels already exist for {subject}_{exp}. Skipping...")
                continue

            # label the predicted curve
            gt_labels = gt_labeler.process(original_curve, mask)
            pred_labels = pred_labeler.process(pred_curve, pred_curve == 0.99)

            # save the labels
            np.savez_compressed(label_output_file, original=gt_labels, pred=pred_labels)

    return gt_labeler.output_folder


def evaluate_metrics(
        subject_exp_dict: dict[str, list[str]],
        exp_config_path: Path,
        dataset_folder: Path,
        curve_output_folder: Path,
        label_output_folder: Path,
        output_folder: Path = None,
        verbose: bool = False,
        is_tflite: bool = False,
        overwrite: bool = False,
):
    """
    Step 3: Evaluate the metrics based on the predicted curves and blink phase labels.

    Args:
        subject_exp_dict: Dictionary of subjects and their corresponding experiments.
        exp_config_path: Path to the experiment configuration file.
        dataset_folder: Path to the processed dataset folder.
        curve_output_folder: Path to the output folder of predicted openness curves.
        label_output_folder: Path to the output folder of blink phase labels.
        output_folder: Path to the output folder of metric results.
        verbose: Whether to print logs.
        is_tflite: Whether to evaluate metrics for the TFLite model or not.
        overwrite: Whether to overwrite the metrics if they already exist.
    """
    evaluator = MetricEvaluator(
        exp_config_path=exp_config_path,
        dataset_folder=dataset_folder,
        output_folder=output_folder,
        verbose=verbose
    )

    metrics_summary_file = evaluator.output_folder / f"metrics_summary{'_tflite' if is_tflite else ''}.pkl"

    if metrics_summary_file.exists() and not overwrite:
        evaluator.logger.info("Metrics already exist. Skipping...")
        with open(metrics_summary_file, "rb") as f:
            summary = pkl.load(f)
    else:
        if not evaluator.output_folder.exists():
            evaluator.output_folder.mkdir(parents=True)

        summary = []
        for subject, exp_list in tqdm(subject_exp_dict.items(), desc="Evaluating metrics"):
            for exp in exp_list:
                # loop through each subject and experiment

                # load the curves and labels
                predicted_curve = np.load(curve_output_folder / f"{subject}_{exp}.npy")
                label_file = label_output_folder / f"{subject}_{exp}{'_tflite' if is_tflite else ''}.npz"

                with np.load(label_file) as data:
                    original_curve_labels = data["original"]
                    predicted_curve_labels = data["pred"]

                original_curve, _ = load_original_curve_and_non_blinking_masks(
                    dataset_folder=dataset_folder,
                    exp_name=exp,
                    dataset_config=evaluator.validation_dataset_config,
                    crop=evaluator.training_config.get_crop_range("validation"),
                )

                # evaluate the metrics
                results = evaluator.process(
                    exp_name=exp,
                    reconstructed_curve=predicted_curve,
                    original_curve=original_curve,
                    reconstructed_curve_labels=predicted_curve_labels,
                    original_curve_labels=original_curve_labels,
                    metrics=["correlation", "rmse_mae", "event", "phase_analysis"]
                )

                results.update({"subject": subject, "experiment": exp})
                summary.append(results.copy())

        # save the metric results
        with open(metrics_summary_file, "wb") as f:
            pkl.dump(summary, f)

    # visualize metric results as tables
    # key results from the paper
    print("\n\n" + "-" * 50)
    print(f"Evaluation Results of BlinkWise-{'Float' if not is_tflite else 'Optimized'} Model @ {evaluator.exp_config.formatted_datetime}")
    print("-" * 50 + "\n")

    print("Curve Prediction Performance (in median):")
    table = PrettyTable()
    table.field_names = ["Correlation", "RMSE", "MAE"]
    correlation = [result["correlation"] for result in summary]
    rmse = [result["rmse"] for result in summary]
    mae = [result["mae"] for result in summary]
    table.add_row([np.median(correlation), np.median(rmse), np.median(mae)])
    print(table)

    print("Blink Detection Performance:")
    table = PrettyTable()
    table.field_names = ["Precision", "Recall"]
    tp = np.array([result["true_positive"] for result in summary])
    fp = np.array([result["false_positive"] for result in summary])
    fn = np.array([result["false_negative"] for result in summary])
    precision = tp.sum() / (tp.sum() + fp.sum())
    recall = tp.sum() / (tp.sum() + fn.sum())
    table.add_row([precision, recall])
    print(table)

    print("Phase Analysis Performance (in median):")
    table = PrettyTable()
    table.field_names = ["Metric", "Closing", "Interphase", "Reopening", "Open"]
    table.add_row(
        ["Abs. Err."] + [
            np.median(np.concatenate([result[phase] for result in summary]))
            for phase in ["closing", "interphase", "reopening", "non_blinking"]
        ]
    )
    table.add_row(
        ["Rel. Err."] + [
            np.median(np.concatenate([result[f"{phase}_rel"] for result in summary]))
            for phase in ["closing", "interphase", "reopening", "non_blinking"]
        ]
    )
    table.add_row(
        ["IoU"] + [
            np.median([result[f"{phase}_iou"] for result in summary])
            for phase in ["closing", "interphase", "reopening", "non_blinking"]
        ]
    )
    print(table)


def main():
    args = parse_arguments()

    result_folder = Path(args.result_folder)
    if not result_folder.exists():
        raise FileNotFoundError(f"Result folder not found: {result_folder}")
    exp_config_file = result_folder / "config.json"
    if not exp_config_file.exists():
        raise FileNotFoundError(f"Experiment configuration file not found: {exp_config_file}")

    exp_config = ExperimentConfig.load(exp_config_file)
    dataset_folder = exp_config.data_folder if args.dataset_folder is None else Path(args.dataset_folder)

    exp_subset, val_exp_subset = prepare_exp_subsets(exp_config)
    subject_exp_dict = summarize_subjects_and_experiments(exp_config.data_folder, exp_subset)

    if args.overwrite_curves:
        if not all([args.overwrite_labels, args.overwrite_metrics]):
            print("Overwriting curves requires overwriting labels and metrics. Setting them to True.")
            args.overwrite_labels = True
            args.overwrite_metrics = True
    if args.overwrite_labels:
        if not args.overwrite_curves:
            print("Overwriting labels requires overwriting metrics. Setting it to True.")
            args.overwrite_metrics = True

    curve_output_folder = predict_curve(
        subject_exp_dict=subject_exp_dict,
        exp_config_path=exp_config_file,
        checkpoint_path=load_checkpoint_path(result_folder, args.tflite),
        dataset_folder=dataset_folder,
        output_folder=args.output_folder,
        verbose=args.verbose,
        overwrite=args.overwrite_curves,
    )

    label_output_folder = label_curve(
        subject_exp_dict=subject_exp_dict,
        exp_config_path=exp_config_file,
        dataset_folder=dataset_folder,
        curve_output_folder=curve_output_folder,
        output_folder=args.output_folder,
        verbose=args.verbose,
        is_tflite=args.tflite,
        overwrite=args.overwrite_labels,
    )

    evaluate_metrics(
        subject_exp_dict=subject_exp_dict,
        exp_config_path=exp_config_file,
        dataset_folder=dataset_folder,
        curve_output_folder=curve_output_folder,
        label_output_folder=label_output_folder,
        output_folder=args.output_folder,
        verbose=args.verbose,
        is_tflite=args.tflite,
        overwrite=args.overwrite_metrics,
    )


if __name__ == "__main__":
    main()
