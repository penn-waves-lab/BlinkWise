from pathlib import Path
from typing import Optional, Union

from .base import AutoLabeler
from .flank import FlankAutoLabeler

def labeler_factory(
        labeler_name: str,
        exp_config_path: Optional[Union[str, Path]] = None,
        peak_quantile=0.85,
        ptp_quantile=0.95,
        closing_wlen=0.3,
        reopening_wlen=0.5,
        quantile_window=30,
        fps=480,
        min_peak_distance=0.2,
        output_folder: Optional[Union[str, Path]] = None,
        verbose=True,
        **kwargs
) -> AutoLabeler:
    if labeler_name == 'flank':
        return FlankAutoLabeler(
            exp_config_path=exp_config_path,
            peak_quantile=peak_quantile,
            ptp_quantile=ptp_quantile,
            closing_wlen=closing_wlen,
            reopening_wlen=reopening_wlen,
            quantile_window=quantile_window,
            fps=fps,
            min_peak_distance=min_peak_distance,
            output_folder=output_folder,
            verbose=verbose,
            **kwargs
        )
    else:
        raise ValueError(f'Unknown labeler name: {labeler_name}')


__all__ = ['labeler_factory']