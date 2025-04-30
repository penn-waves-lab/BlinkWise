import argparse
import atexit
import json
import sys
from pathlib import Path
from typing import Sequence

script_dir = Path(__file__).resolve().parent
project_folder = script_dir.parent
if str(project_folder) not in sys.path:
    sys.path.insert(0, str(project_folder))

from src.models.config import ExperimentConfig
from src.models.training import TranslatorTrainer
from src.models.utils import prepare_exp_subsets


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment configuration")
    parser.add_argument(
        "-c", "--config", help="Path to configuration JSON file", default="../configs/example_config.json",
    )
    parser.add_argument(
        "-s", "--config-string", help="A string of JSON file", default=""
    )
    return parser.parse_args()


def load_exp_config(config_file_path, config_string) -> ExperimentConfig:
    """
    Load experiment configuration from a JSON file or a JSON string.

    Args:
        config_file_path: Path to the JSON file containing the experiment configuration
        config_string: A JSON string containing the experiment configuration

    Returns:

    """
    config_file_path = Path(config_file_path)
    config = None
    # Attempt to load configuration from JSON string
    if config_string:
        try:
            config_dict = json.loads(config_string)
            config = ExperimentConfig.load_from_dict(config_dict)
            print("Parsed JSON data from string:", config)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON string: {e}")

    # If JSON string is invalid or not provided, try to load from file
    if config is None:
        if config_file_path.exists():
            try:
                config = ExperimentConfig.load(config_file_path)
                print("Loaded JSON data from file:", config)
            except Exception as e:
                raise ValueError(f"Failed to load JSON from file {config_file_path}: {e}")
        else:
            raise FileNotFoundError(f"Cannot find the provided config file at {config_file_path}")
    return config


def create_trainer(config: ExperimentConfig, exp_subset: Sequence[str], val_exp_subset: Sequence[str] = None):
    """
    Create a trainer object based on the model type specified in the configuration.

    Args:
        config: Experiment configuration
        exp_subset: A list of experiments to include in the training set
        val_exp_subset: A list of experiments to include in the validation set

    Returns:
        A trainer object
    """
    model_type = config.model_config.__class__.__name__
    if model_type == "UNetConfig":
        return TranslatorTrainer(config, exp_subset, val_exp_subset)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main():
    args = parse_arguments()
    exp_config = load_exp_config(args.config, args.config_string)
    exp_subset, val_exp_subset = prepare_exp_subsets(exp_config)

    print(f"Experiment subset size: {len(exp_subset)}; Validation subset size: {len(val_exp_subset)}")

    trainer = create_trainer(exp_config, exp_subset, val_exp_subset)
    atexit.register(trainer.cleanup)
    trainer.train()


if __name__ == "__main__":
    main()
