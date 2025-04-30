from abc import ABC, abstractmethod
import json
import os
import platform
from typing import Sequence, cast, Literal

import tensorflow as tf
if os.environ.get("BLINKWISE_USE_TFMOT_KERAS", "0") == "0":
    from tensorflow import keras
else:
    from tensorflow_model_optimization.python.core.keras.compat import keras

from .callbacks import visualizer_factory, WeightScheduler
from .losses import loss_factory, loss_fn_names_to_list_of_funcs
from .metrics import MaskedCorrelationMetric

from ..config import BlinkDatasetConfig, ExperimentConfig, UNetConfig
from ..data.dataset import AllExperimentDataset
from ..utils import convert_history_to_python_types, get_first_experiment_per_subject


def exp_scheduler(epoch, lr):
    if epoch < 40:
        return lr
    elif epoch % 20 == 0:
        return lr * 0.1
    else:
        return lr


def prepare_lr_scheduler(lr_scheduler: Literal["custom", "plateau"]) -> list[keras.callbacks.Callback]:
    if lr_scheduler == "custom":
        return [keras.callbacks.LearningRateScheduler(exp_scheduler)]
    elif lr_scheduler == "plateau":
        return [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)]


class BaseTrainer(ABC):
    def __init__(self, config: ExperimentConfig, exp_subset: Sequence[str] = (), val_exp_subset: Sequence[str] = ()):
        self.config = config
        self.exp_subset = exp_subset
        self.val_exp_subset = val_exp_subset

        # configurations
        self.training_dataset_config = self.config.dataset_configs[0]
        self.val_dataset_config = self.config.dataset_configs[1]
        self.training_config = self.config.training_config
        self.model_config = self.config.model_config

        # datasets
        self.all_exp_ds = None
        self.val_all_exp_ds = None

        self.training_dataset = None
        self.val_dataset = None

        self.training_viz_subset = None
        self.val_viz_subset = None

        # models
        self.model = None

        self.logger = tf.get_logger()

    @abstractmethod
    def prepare_model(self):
        pass

    def summarize_model(self):
        if self.model is not None:
            try:
                keras.utils.plot_model(
                    self.model,
                    to_file=os.path.join(self.config.trial_output_folder, "model.png"),
                    show_shapes=True, rankdir="TB", expand_nested=True
                )
            except Exception as e:
                self.logger.warn(f"Model architecture visualization failed with {e}. This is likely due to a missing graphviz installation.")
            self.model.summary()

    def prepare_dataset(self):
        # datasets for training and validation
        #     train dataset
        self.all_exp_ds = AllExperimentDataset(
            self.config.data_folder,
            self.training_dataset_config,
            exp_subset=self.exp_subset
        )
        train_crop = self.config.training_config.get_crop_range("train")
        self.training_dataset = self.all_exp_ds.as_dataset(crop=train_crop)

        #     validation dataset
        self.val_all_exp_ds = AllExperimentDataset(
            self.config.data_folder,
            self.val_dataset_config,
            exp_subset=self.val_exp_subset
        )
        val_crop = self.config.training_config.get_crop_range("validation")
        self.val_dataset = self.val_all_exp_ds.as_dataset(crop=val_crop)

        input_signature, label_signature = self.all_exp_ds.get_output_signature()
        input_shape = input_signature.shape
        self.logger.info(f"{input_shape=}")

    def setup_callbacks(self):
        cbs = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.config.checkpoint_path, f"{self.config.model_name}.keras"),
                save_best_only=True
            ),
        ]
        return cbs

    def train(self):
        if not self.config.full_config_path.exists():
            self.config.materialize()

        self.prepare_dataset()
        self.prepare_model()
        callbacks = self.setup_callbacks()

        history = self.model.fit(
            self.training_dataset,
            epochs=self.config.training_config.epochs,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=2,
        )

        self.save_history(history)

    def save_history(self, history):
        with open(os.path.join(self.config.trial_output_folder, "history.json"), "w") as fp:
            json.dump(convert_history_to_python_types(history.history), fp)

    def cleanup(self):
        if self.all_exp_ds is not None:
            self.all_exp_ds.cleanup()
        if self.val_all_exp_ds is not None:
            self.val_all_exp_ds.cleanup()


class TranslatorTrainer(BaseTrainer):
    def __init__(self, config: ExperimentConfig, exp_subset: Sequence[str] = (), val_exp_subset: Sequence[str] = ()):
        super().__init__(config, exp_subset, val_exp_subset)

        self.training_dataset_config = cast(BlinkDatasetConfig, self.training_dataset_config)
        self.val_dataset_config = cast(BlinkDatasetConfig, self.val_dataset_config)

    def prepare_dataset(self):
        super().prepare_dataset()

        # sample a subset of dataset for visualization
        train_crop = self.config.training_config.get_crop_range("train")
        val_crop = self.config.training_config.get_crop_range("validation")

        # training set visualization
        exps_to_viz = list(get_first_experiment_per_subject(self.config.data_folder, self.exp_subset).values())
        viz_subset_ds = AllExperimentDataset(
            self.config.data_folder,
            self.training_dataset_config,
            exp_subset=exps_to_viz
        )
        self.training_viz_subset = viz_subset_ds.sample_subset(
            crop=train_crop, n_samples=4,
            **{f"{t}_n_samples": 1 for t in self.training_dataset_config.sample_non_blinks}
        )

        # validation set visualization
        val_exps_to_viz = list(get_first_experiment_per_subject(self.config.data_folder, self.val_exp_subset).values())
        val_viz_subset_ds = AllExperimentDataset(
            self.config.data_folder,
            self.val_dataset_config,
            exp_subset=val_exps_to_viz
        )
        self.val_viz_subset = val_viz_subset_ds.sample_subset(
            crop=val_crop, n_samples=2,
            **{f"{t}_n_samples": 1 for t in self.val_dataset_config.sample_non_blinks}
        )

    def prepare_model(self):
        input_shape = self.training_dataset.element_spec[0].shape
        if self.config.model_config.__class__.__name__ == "UNetConfig":
            from ..networks import unet_factory
            self.model_config = cast(UNetConfig, self.model_config)
            self.model = unet_factory(input_shape[1:], self.model_config)
        else:
            raise ValueError(f"Undefined model config class name {self.model_config.__class__.__name__}.")

        loss_weights_variables = self.training_config.loss_weights_as_variables

        # use legacy adam optimizer for macOS
        if platform.system() == "Darwin":
            optimizer = keras.optimizers.legacy.Adam(self.training_config.init_lr)
        else:
            optimizer = keras.optimizers.Adam(self.training_config.init_lr)

        self.model.compile(
            optimizer=optimizer,
            loss=loss_factory(
                loss_fn_names=self.config.training_config.loss_fn_names,
                loss_params=self.config.training_config.loss_params,
                loss_weights=loss_weights_variables
            ),
            metrics=[
                        MaskedCorrelationMetric()
                    ] + loss_fn_names_to_list_of_funcs(
                loss_fn_names=self.training_config.loss_fn_names,
                loss_params=self.training_config.loss_params,
            ),
        )

        self.summarize_model()

    def setup_callbacks(self):
        cbs = super().setup_callbacks()
        cbs += [
            visualizer_factory(
                visualizer_type="unet",
                log_dir=self.config.summary_path.as_posix(),
                data_subset=self.training_viz_subset,
                freq=5,
                tag='train',
            ),
            visualizer_factory(
                visualizer_type="unet",
                log_dir=self.config.summary_path.as_posix(),
                data_subset=self.val_viz_subset,
                freq=5,
                tag='val',
            ),
        ]
        if self.training_config.lr_scheduler is not None:
            cbs += prepare_lr_scheduler(self.training_config.lr_scheduler)
        # loss weights scheduler
        if self.training_config.loss_weight_scheduler == "custom":
            cbs += [WeightScheduler(self.training_config.loss_weights_as_variables)]
        # added last to also record lr change.
        cbs += [keras.callbacks.TensorBoard(log_dir=self.config.summary_path)]
        return cbs
