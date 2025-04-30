from pathlib import Path
from typing import Union, Literal

from src.core import validate_literal_args
from .base import CurveReconstructor
from .keras_model import KerasCurveReconstructor
from .tflite_model import TFLiteCurveReconstructor

@validate_literal_args
def curve_reconstructor_factory(
        exp_config_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        dataset_folder: Union[str, Path] = None,
        output_folder: Union[str, Path] = None,
        verbose: bool = True,
) -> CurveReconstructor:
    model_type = Path(checkpoint_path).suffix[1:]
    if model_type == "keras":
        return KerasCurveReconstructor(
            exp_config_path=exp_config_path,
            checkpoint_path=checkpoint_path,
            dataset_folder=dataset_folder,
            output_folder=output_folder,
            verbose=verbose
        )
    elif model_type == "tflite":
        return TFLiteCurveReconstructor(
            exp_config_path=exp_config_path,
            checkpoint_path=checkpoint_path,
            dataset_folder=dataset_folder,
            output_folder=output_folder,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

__all__ = ["curve_reconstructor_factory"]