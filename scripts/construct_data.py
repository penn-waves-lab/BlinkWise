import argparse
import json
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_folder = script_dir.parent
if str(project_folder) not in sys.path:
    sys.path.insert(0, str(project_folder))

from src.data.dataset import construct_dataset, ProcessingConfig

def main():
    parser = argparse.ArgumentParser(description="Construct a dataset for the openness prediction model.")

    # General arguments
    general_group = parser.add_argument_group("General Arguments")
    general_group.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="Path to the root folder containing the experiment data.",
    )
    general_group.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the output folder where the dataset will be saved.",
    )
    general_group.add_argument(
        "--experiment-dates",
        nargs="*",
        default=None,
        help="List of experiment dates to process. If not provided, all experiments will be processed.",
    )
    general_group.add_argument(
        "--processing-protocol",
        nargs="+",
        default=["range-querying-fft", "low-pass-filtering", "diff", "normalization"],
        help="Processing protocol to apply to the radar data.",
    )
    general_group.add_argument(
        "--append-protocol-suffix-to-output-folder",
        action="store_true",
        help="Whether to append the processing protocol suffix to the output folder name.",
    )
    general_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Whether to print logs.",
    )
    general_group.add_argument(
        "--processing-config",
        type=str,
        help="Processing configuration as a JSON string or a path to a JSON file.",
    )

    # ProcessingConfig arguments
    processing_group = parser.add_argument_group(
        "ProcessingConfig Arguments",
        description="When processing config is not provided, these arguments will be used to construct the ProcessingConfig."
    )
    processing_group.add_argument(
        "--min-range", type=float, help="Minimum range for range querying. In centimeters.", default=2.5
    )
    processing_group.add_argument(
        "--max-range", type=float, help="Maximum range for range querying. In centimeters.", default=5.5
    )
    processing_group.add_argument(
        "--n-range-bins", type=int, help="Number of range bins between min and max ranges.", default=10
    )
    processing_group.add_argument(
        "--stat-window-size", type=int, help="Size of the statistical window. In seconds.", default=30
    )
    processing_group.add_argument(
        "--stat-hop", type=int, help="Hop size for statistical windowing. In seconds.", default=10
    )
    processing_group.add_argument(
        "--norm-stat",
        type=str,
        choices=["std", "max"],
        help="Normalization statistic to use.",
        default="max"
    )

    args = parser.parse_args()

    # Determine how to construct the ProcessingConfig
    if args.processing_config:
        if args.processing_config.endswith(".json"):
            processing_config = ProcessingConfig.load_from_json(args.processing_config)
        else:
            config_dict = json.loads(args.processing_config)
            processing_config = ProcessingConfig(**config_dict)
    else:
        processing_config = ProcessingConfig(
            min_range=args.min_range,
            max_range=args.max_range,
            n_range_bins=args.n_range_bins,
            stat_window_size=args.stat_window_size,
            stat_hop=args.stat_hop,
            norm_stat=args.norm_stat,
        )

    # Call the construct_dataset function
    construct_dataset(
        data_folder=Path(args.data_folder),
        output_folder=Path(args.output_folder),
        experiment_dates=args.experiment_dates,
        processing_protocol=tuple(args.processing_protocol),
        processing_config=processing_config,
        append_protocol_suffix_to_output_folder=args.append_protocol_suffix_to_output_folder,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
