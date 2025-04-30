from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class OutputPathConfig:
    output_folder: Optional[Path] = None
    reconstructed_curve_output_folder: Optional[Path] = None
    label_output_folder: Optional[Path] = None
    metric_output_folder: Optional[Path] = None
    _instance = None

    @classmethod
    def init(cls,
             output_folder: Optional[Union[str, Path]] = None,
             reconstructed_curve_output_folder: Optional[Union[str, Path]] = None,
             label_output_folder: Optional[Union[str, Path]] = None,
             metric_output_folder: Optional[Union[str, Path]] = None) -> 'OutputPathConfig':
        """Initialize or update the path configuration."""
        if cls._instance is None:
            cls._instance = cls()

        if output_folder is not None:
            cls._instance.output_folder = Path(output_folder)

        if reconstructed_curve_output_folder is not None:
            cls._instance.reconstructed_curve_output_folder = Path(reconstructed_curve_output_folder)
        else:
            cls._instance.reconstructed_curve_output_folder = cls._instance.output_folder / "reconstructed_curves"

        if label_output_folder is not None:
            cls._instance.label_output_folder = Path(label_output_folder)
        else:
            cls._instance.label_output_folder = cls._instance.output_folder / "labels"

        if metric_output_folder is not None:
            cls._instance.metric_output_folder = metric_output_folder
        else:
            cls._instance.metric_output_folder = cls._instance.output_folder / "metrics"

        return cls._instance

    @classmethod
    def get_instance(cls) -> 'OutputPathConfig':
        """Get the singleton instance of the path configuration."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance of the path configuration."""
        cls._instance = None
