import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional

from src.core import setup_logging
from src.models.config import ExperimentConfig


class BaseProcessor(ABC):
    """Base class for all processors."""

    def __init__(
            self,
            name: str,
            exp_config_path: Union[str, Path],
            output_folder: Optional[Union[str, Path]] = None,
            verbose: bool = True
    ):
        self.name = name

        self.exp_config_path = Path(exp_config_path)
        self.exp_config = self._load_config()

        self.output_folder = (self.exp_config.trial_output_folder / self.name) if output_folder is None else (Path(output_folder) / self.name)
        self.verbose = verbose

        self._setup_logging()

    @property
    def training_dataset_config(self):
        return self.exp_config.dataset_configs[0]

    @property
    def validation_dataset_config(self):
        return self.exp_config.dataset_configs[1]

    @property
    def training_config(self):
        return self.exp_config.training_config

    @property
    def model_config(self):
        return self.exp_config.model_config

    def _load_config(self) -> ExperimentConfig:
        if not self.exp_config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.exp_config_path}")
        with open(self.exp_config_path, "r") as f:
            exp_config_dict = json.load(f)

        exp_config = ExperimentConfig.load_from_dict(exp_config_dict)
        return exp_config

    def _setup_logging(self):
        log_path = None
        if self.verbose and self.output_folder is not None:
            log_path = self.output_folder / f"{self.name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)

        setup_logging(log_path, logging.INFO if self.verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def process(self, *args, **kwargs):
        """
        The main interface all subclass processors should implement.
        """
        pass
