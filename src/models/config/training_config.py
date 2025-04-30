from typing import Literal, Sequence, Optional

import tensorflow as tf

from .base_config import BaseConfig
from src.core import validate_literal_args


class TrainingConfig(BaseConfig):
    @validate_literal_args
    def __init__(
            self,
            train_crop: tuple[float, float] = (0., 0.9),
            valid_crop: tuple[float, float] = (0.9, 1.),
            epochs=50,
            init_lr=1e-2,
            lr_scheduler: Optional[Literal["custom", "plateau"]] = None,
            loss_weight_scheduler: Optional[Literal["custom"]] = None,
            loss_fn_names: Sequence[str] = ("masked_mse",),
            loss_weights: Sequence[float] = None,
            loss_params: dict = None,
    ):
        """
        Training configuration for the model.

        Args:
            train_crop: Crop range for training data. The range is [start, end). In percentage. Default is (0., 0.9).
            valid_crop: Crop range for validation data. The range is [start, end). In percentage. Default is (0.9, 1.).
            epochs: Number of epochs to train the model. Default is 50.
            init_lr: Initial learning rate. Default is 1e-2.
            lr_scheduler: Learning rate scheduler. Default is None.
            loss_weight_scheduler: Loss weight scheduler. Default is None.
            loss_fn_names: Names of the loss functions to use. Default is ("masked_mse",).
            loss_weights: Weights for the loss functions. Default is None.
            loss_params: Parameters for the loss functions. Default is None.
        """
        super().__init__()

        self.train_crop = train_crop
        self.valid_crop = valid_crop

        self.epochs = epochs

        # learning rate related
        self.init_lr = init_lr
        self.lr_scheduler = lr_scheduler

        # loss related
        #   loss weight scheduling
        self.loss_weight_scheduler = loss_weight_scheduler
        #   initialize loss weights
        if loss_weights is None:
            # Uniform weights
            loss_weights = [1] * len(loss_fn_names)
        elif len(loss_weights) != len(loss_fn_names):
            raise ValueError("The length of loss_weights must be equal to the number of specified loss function names.")

        self.loss_fn_names = loss_fn_names
        self.loss_weights = loss_weights
        if loss_params is None:
            loss_params = {}
        self.loss_params = loss_params

    @property
    def loss_weights_as_variables(self) -> Sequence[tf.Variable]:
        return [tf.Variable(initial_value=w, dtype=tf.float32) for w in self.loss_weights]

    def get_crop_range(self, dataset_type: Literal["train", "validation"]) -> tuple[float, float]:
        if dataset_type == "train":
            return self.train_crop
        elif dataset_type == "validation":
            return self.valid_crop
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Supported: train, and validation.")
