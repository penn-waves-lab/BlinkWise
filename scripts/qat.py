import os
os.environ["BLINKWISE_USE_TFMOT_KERAS"] = "1"

import atexit
import datetime
import json
import argparse
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_folder = script_dir.parent
if str(project_folder) not in sys.path:
    sys.path.insert(0, str(project_folder))

from src.models.config import ExperimentConfig
from src.models.utils import prepare_exp_subsets
from src.optimization.qat_trainer import QATTranslatorTrainer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Quantization-aware training")
    parser.add_argument(
        "-r", "--result-folder", help="Path to the result folder", required=True,
    )
    parser.add_argument(
        "-o", "--output-folder", help="Path to the output folder", default=None,
    )
    parser.add_argument(
        "--finetune-lr", help="Learning rate for fine-tuning", type=float, default=5e-4,
    )
    parser.add_argument(
        "--finetune-epochs", help="Number of epochs for fine-tuning", type=float, default=2.7,
    )
    parser.add_argument(
        "--overwrite", help="Overwrite existing tflite model", action="store_true",
    )
    return parser.parse_args()


def qat(
        result_folder: Path,
        output_folder: Path,
        finetune_lr=5e-4,
        finetune_epochs=2.7,
        overwrite=False,
):
    """
    Perform quantization-aware training and convert the model to tflite format.

    Args:
        result_folder: Path to the folder containing the trained floating-point model
        output_folder: Path to the folder to save the quantized tflite model
        finetune_lr: Learning rate for fine-tuning
        finetune_epochs: Number of epochs for fine-tuning. Can be a float to indicate a fraction of an epoch
        overwrite: Overwrite existing tflite model
    """
    original_model_path = result_folder / "checkpoint" / "unet.keras"
    tflite_model_path = output_folder / "qat" / f"lr={finetune_lr}-epochs={finetune_epochs}" / "quantized_model.tflite"
    if not tflite_model_path.parent.exists():
        tflite_model_path.parent.mkdir(parents=True)
    elif tflite_model_path.exists() and not overwrite:
        print(f"Tflite model already exists at {tflite_model_path}.")
        return

    exp_config_path = result_folder / "config.json"
    if not exp_config_path.exists():
        raise FileNotFoundError(f"Experiment config not found at {exp_config_path}")
    with open(exp_config_path, "r") as f:
        exp_config_dict = json.load(f)
    exp_config = ExperimentConfig.load_from_dict(exp_config_dict)
    exp_subset, val_exp_subset = prepare_exp_subsets(exp_config)

    trainer = QATTranslatorTrainer(
        config=exp_config,
        checkpoint_path=original_model_path,
        fine_tune_lr=finetune_lr,
        fine_tune_epochs=finetune_epochs,
        exp_subset=exp_subset,
        val_exp_subset=val_exp_subset,
    )
    atexit.register(trainer.cleanup)
    trainer.train()
    tflite_model = trainer.tflite_conversion()

    # save the tflite model
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Q-aware tflite model saved to {tflite_model_path} @ {datetime.datetime.now()}")


def main():
    args = parse_arguments()
    qat(
        Path(args.result_folder),
        Path(args.output_folder) if args.output_folder else Path(args.result_folder),
        finetune_lr=args.finetune_lr,
        finetune_epochs=args.finetune_epochs,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
