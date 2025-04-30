from typing import Literal, Sequence

from src.core import validate_literal_args
from src.data.dataset.dataset_specification import ProcessingProtocol, BlinkParameter
from .base_config import BaseConfig

MODALITIES = Literal["real-imaginary", "magnitude-angle", "magnitude"]
SAMPLER_TYPES = Literal["blink"]
EVENT_DETECTORS = Literal["video", "fsm"]
RESIZING_METHODS = Literal["resize", "pad", "as-is"]


class BaseDatasetConfig(BaseConfig):
    @validate_literal_args
    def __init__(
            self,
            dataset_processing_protocol: ProcessingProtocol = (
                    "range-querying-fft", "low-pass-filtering", "diff", "normalization"),
            onsite_processing_protocol=("mean-beamforming", "select-bins-5"),
            modality: MODALITIES = "magnitude-angle",
            sampler_type: SAMPLER_TYPES = "blink",
            event_detector: EVENT_DETECTORS = "video",
            shuffle: bool = True,
            normalize_window: int = 30000,
            batch_size: int = 32,
            seed: int = 1234
    ):
        super().__init__()

        if modality == "signed-magnitude" and not any(map(lambda c: "project" in c, dataset_processing_protocol)):
            raise ValueError("Signed magnitude is only supported when projected diff is specified in the protocol.")

        if not modality == "signed-magnitude" and any(map(lambda c: "project" in c, dataset_processing_protocol)):
            raise ValueError(f"Use signed-magnitude instead of {modality} with projected diff in the protocol.")

        self.dataset_processing_protocol = dataset_processing_protocol
        self.onsite_processing_protocol = onsite_processing_protocol
        self.modality = modality
        self.sampler_type = sampler_type
        self.event_detector = event_detector

        self.shuffle = shuffle

        self.normalize_window = int(normalize_window)
        self.batch_size = batch_size
        self.seed = seed

    @property
    def use_doppler(self) -> bool:
        return any(map(lambda c: "crop" not in c and "doppler" in c, self.dataset_processing_protocol))


class BlinkDatasetConfig(BaseDatasetConfig):
    @validate_literal_args
    def __init__(
            self,
            # dataset specific configs
            uniform_length=1024,
            resizing_method: RESIZING_METHODS = "pad",
            random_margin=32,
            sample_non_blinks: Sequence[Literal["event", "event-no-expand", "negative"]] = (),
            curve_name: BlinkParameter = "blink_ratio",
            curve_processing_protocol: Sequence[Literal["abs", "shift", "scale"]] = ("abs", "scale"),
            augmentation: Sequence[Literal["left", "right"]] = (),
            augmentation_ratio: float = 0.5,
            # general config for all datasets
            dataset_processing_protocol: ProcessingProtocol = (
                    "range-querying-fft", "low-pass-filtering", "diff", "normalization"),
            onsite_processing_protocol=("mean-beamforming", "select-bins-5"),
            modality: MODALITIES = "magnitude-angle",
            sampler_type: SAMPLER_TYPES = "blink",
            event_detector: EVENT_DETECTORS = "fsm",
            shuffle: bool = True,
            normalize_window: int = 30000,
            batch_size: int = 32,
            seed: int = 1234,
    ):
        if not sampler_type == "blink":
            raise ValueError(f"Only blink sampler is supported for BlinkDatasetConfig. Got {sampler_type}.")

        if resizing_method == "as-is" and not batch_size == 1:
            raise ValueError("When no resizing (as-is) is toggled, batching is disabled. Please set batch size to 1.")

        super().__init__(
            dataset_processing_protocol=dataset_processing_protocol,
            onsite_processing_protocol=onsite_processing_protocol,
            modality=modality,
            sampler_type=sampler_type,
            event_detector=event_detector,
            shuffle=shuffle,
            normalize_window=normalize_window,
            batch_size=batch_size,
            seed=seed,
        )

        self.uniform_length = uniform_length
        self.resizing_method = resizing_method
        self.random_margin = random_margin
        self.sample_non_blinks = sample_non_blinks
        self.curve_name = curve_name
        self.curve_processing_protocol = list(set(curve_processing_protocol))

        self.event_detector = event_detector
        self.augmentation = list(set(augmentation))
        self.augmentation_ratio = augmentation_ratio

    @property
    def n_range_bins(self):
        bin_selection_op = list(filter(lambda c: "select-bins" in c, self.onsite_processing_protocol))
        if len(bin_selection_op) == 0:
            # TODO: make this adaptive if the underlying dataset is not queried at 10 bins
            return 10
        else:
            return int(bin_selection_op[0].split("-")[-1])
