import json
from datetime import datetime
from pathlib import Path
from typing import Sequence, Union, Optional

from src.core import validate_literal_args
from .base_config import BaseConfig, dict_to_markdown_table
from .dataset_config import BaseDatasetConfig, BlinkDatasetConfig
from .model_config import BaseModelConfig, UNetConfig
from .training_config import TrainingConfig


class ExperimentConfig(BaseConfig):
    """
    a configuration class that contains all information about the experiment

    training artifacts organization:

    output_root_folder/
    └── experiment_name/
        └── model_name_YYYYMMDD_HHMMSS/ (one trial)
            ├── checkpoint/
            ├── summary/
            │   ├── intermediate image outputs
            │   ├── train/
            │   └── validation/
            ├── config_summary.md
            ├── history.json
            └── config.json
    """

    @validate_literal_args
    def __init__(
            self,
            experiment_name: str,
            trial_name: str,
            data_folder: Union[str, Path],
            training_config: TrainingConfig,
            model_config: BaseModelConfig,
            dataset_configs: Union[BaseDatasetConfig, list[BaseDatasetConfig]],
            output_root_folder: Union[str, Path] = "",
            model_name="",
            note="",
            subjects_to_exclude: Optional[Union[str, Sequence[str]]] = (),
            leave_one_out_subject: Optional[str] = None,
            formatted_datetime=None,
    ):
        """
        Args:
            experiment_name: Name of the experiment. It is used to create a folder to save all trials of the experiment.
            trial_name: Name of the trial. Under one experiment, trials are different runs of various configurations.
            data_folder: Path to the folder containing the processed dataset.
            training_config: Configuration for training the model.
            model_config: Configuration for the model to be trained.
            dataset_configs: Configuration for the dataset.
            output_root_folder: Path to the root folder to save all experiments. Default is the current working directory.
            model_name: A nickname for the model.
            note: A more detailed description of the trial. Usually notes how the trial is different from others.
            subjects_to_exclude: A list of subjects to exclude from the dataset.
            leave_one_out_subject: A subject to leave out from the dataset for validation.
            formatted_datetime: A formatted datetime string to be used as a suffix for the trial folder.
        """
        super().__init__()

        # prepare folders to save artifacts
        if formatted_datetime is None:
            formatted_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.formatted_datetime = formatted_datetime
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.data_folder = Path(data_folder)
        self.output_root_folder = Path(output_root_folder)
        self.model_name = model_name
        self.note = note

        if isinstance(subjects_to_exclude, str):
            subjects_to_exclude = [subjects_to_exclude]
        self.subjects_to_exclude = subjects_to_exclude
        self.leave_one_out_subject = leave_one_out_subject

        self.output_folder = self.output_root_folder / self.experiment_name
        if not self.model_name:
            self.trial_output_folder = self.output_folder / self.formatted_datetime
        else:
            self.trial_output_folder = self.output_folder / f"{self.model_name}_{self.formatted_datetime}"
        self.summary_path = self.trial_output_folder / "summary"
        self.checkpoint_path = self.trial_output_folder / "checkpoint"

        # to save the config in a Markdown format that is more readable for human
        self.config_summary_path = self.trial_output_folder / "config_summary.md"
        # to save configurations
        self.full_config_path = self.trial_output_folder / "config.json"

        if isinstance(dataset_configs, BaseConfig):
            dataset_configs = [dataset_configs]
        self.training_config = training_config
        self.dataset_configs = dataset_configs
        self.model_config = model_config

    def materialize(self):
        if not self.output_folder.exists():
            self.output_folder.mkdir(parents=True)
        if not self.summary_path.exists():
            self.summary_path.mkdir(parents=True)
        if not self.checkpoint_path.exists():
            self.checkpoint_path.mkdir(parents=True)

        # save configurations
        #   write the experiment note
        config_summary = []
        if self.note:
            config_summary.append(f"{self.trial_name}: {self.note}")
        else:
            config_summary.append("No note provided.")
        #   write configurations about datasets
        config_summary.append("Dataset Configs")
        for c in self.dataset_configs:
            config_summary.append(dict_to_markdown_table(c.to_dict()))
        #   write configs about the model to be trained
        config_summary.append("Model Configs")
        config_summary.append(dict_to_markdown_table(self.model_config.to_dict()))
        #   write configs about the training configuration
        config_summary.append("Training Configs")
        config_summary.append(dict_to_markdown_table(self.training_config.to_dict()))
        #   write down the file
        with open(self.config_summary_path, 'w') as f:
            f.write("\n\n".join(config_summary))

        # save configurations
        self.save(self.full_config_path)

    def save(self, json_file):
        to_save = {
            "experiment_name": self.experiment_name,
            "trial_name": self.trial_name,
            "data_folder": self.data_folder.as_posix(),
            "output_root_folder": self.output_root_folder.as_posix(),
            "model_name": self.model_name,
            "note": self.note,
            "subjects_to_exclude": self.subjects_to_exclude,
            "leave_one_out_subject": self.leave_one_out_subject,
            "formatted_datetime": self.formatted_datetime,
            "training_config": {
                "class_name": self.training_config.__class__.__name__,
                "data": self.training_config.to_dict()
            },
            "model_config": {
                "class_name": self.model_config.__class__.__name__,
                "data": self.model_config.to_dict()
            },
            "dataset_configs": [{
                "class_name": c.__class__.__name__,
                "data": c.to_dict()
            } for c in self.dataset_configs],
        }
        with open(json_file, 'w') as config_file:
            json.dump(to_save, config_file, indent=4)

    @classmethod
    def load(cls, json_file):
        with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)
        return cls.load_from_dict(config_dict)

    @classmethod
    def load_from_dict(cls, config_dict: dict):
        kwargs = cls._construct_kwargs(config_dict)
        return cls(**kwargs)

    @staticmethod
    def _construct_kwargs(config_dict):
        kwargs = {}
        for k, v in config_dict.items():
            if "config" not in k:
                kwargs[k] = v
            else:
                if isinstance(v, list):
                    constructed_configs = []
                    for v_elem in v:
                        class_name = v_elem["class_name"]
                        constructed_configs.append(
                            globals()[class_name](**v_elem["data"])
                        )
                    kwargs[k] = constructed_configs
                else:
                    class_name = v["class_name"]
                    kwargs[k] = globals()[class_name](**v["data"])
        return kwargs
