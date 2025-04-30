import os
import platform
from pathlib import Path
from typing import Sequence, cast, Union

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from .qat_utils import ParameterTransfer1Dto2D, GroupNormQuantizeConfig
from ..models.config import ExperimentConfig, BlinkDatasetConfig, UNetConfig
from ..models.data import AllExperimentDataset
from ..models.networks import unet_factory
from ..models.training.losses import loss_factory
from ..models.training.metrics import MaskedCorrelationMetric
from ..models.training.trainer import BaseTrainer

if os.environ.get("BLINKWISE_USE_TFMOT_KERAS", "0") == "0":
    from tensorflow import keras
else:
    from tensorflow_model_optimization.python.core.keras.compat import keras


class QATTranslatorTrainer(BaseTrainer):
    """
    Trainer for quantization-aware training of the translator model.
    """

    def __init__(
            self,
            config: ExperimentConfig,
            checkpoint_path: Union[str, Path],
            fine_tune_lr: float = 1e-4,
            fine_tune_epochs: Union[int, float] = 10,
            exp_subset: Sequence[str] = (),
            val_exp_subset: Sequence[str] = (),
    ):
        super().__init__(config, exp_subset, val_exp_subset)

        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path = checkpoint_path
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
        if self.checkpoint_path.suffix != ".keras":
            raise ValueError(f"Invalid checkpoint file extension: {self.checkpoint_path.suffix}")

        self.fine_tune_lr = fine_tune_lr
        self.fine_tune_epochs = fine_tune_epochs

        self.training_dataset_config = cast(BlinkDatasetConfig, self.training_dataset_config)
        self.val_dataset_config = cast(BlinkDatasetConfig, self.val_dataset_config)

    def _transfer_parameters(self, model_1d: keras.models.Model, model_2d: keras.models.Model):
        stats = ParameterTransfer1Dto2D.transfer_parameters(model_1d, model_2d)

        self.logger.info("Parameter transfer statistics:")
        self.logger.info(f"- Matched layers: {stats['matched_layers']}")
        self.logger.info(f"- Successfully transferred: {stats['transferred_layers']}")
        if stats["skipped_layers"]:
            self.logger.info("\nSkipped layers:")
            for skip in stats["skipped_layers"]:
                self.logger.info(f"- {skip['name']}: {skip['reason']}")

    def fix_model_batch_size(self) -> keras.models.Model:
        input_info = self.model.input
        batch_size_1_input = keras.Input(
            shape=input_info.shape[1:], batch_size=1, dtype=input_info.dtype
        )
        batch_size_1_model = keras.models.Model(inputs=batch_size_1_input, outputs=self.model(batch_size_1_input), name="fixed_model")
        return batch_size_1_model

    def tflite_conversion(self) -> bytes:
        def rep_gen():
            training_viz_subset = self.all_exp_ds.sample_subset(
                crop=self.training_config.train_crop,
                n_samples=20,
                **{f"{t}_n_samples": 1 for t in self.training_dataset_config.sample_non_blinks},
            )
            ds = tf.data.Dataset.from_tensor_slices(
                training_viz_subset[0].astype(np.float32)
            ).batch(1)
            for x in ds:
                yield [x]

        batch_size_1_model = self.fix_model_batch_size()
        converter = tf.lite.TFLiteConverter.from_keras_model(batch_size_1_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()
        return tflite_model

    def prepare_dataset(self):
        self.all_exp_ds = AllExperimentDataset(
            self.config.data_folder,
            self.training_dataset_config,
            exp_subset=self.exp_subset
        )
        train_crop = self.config.training_config.get_crop_range("train")
        self.training_dataset = self.all_exp_ds.as_dataset(crop=train_crop, return_dataset_directly=True)

    def prepare_model(self):
        input_shape = self.training_dataset.element_spec[0].shape

        original_1d_model = keras.models.load_model(self.checkpoint_path, compile=False)

        lifted_2d_model = unet_factory(
            input_shape,
            model_config=cast(UNetConfig, self.model_config),
            lift_to_2d=True,
        )
        self._transfer_parameters(model_1d=original_1d_model, model_2d=lifted_2d_model)

        # qat annotation
        quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
        quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
        quantize_scope = tfmot.quantization.keras.quantize_scope
        quantize_apply = tfmot.quantization.keras.quantize_apply

        def apply_quantization_to_gn(layer):
            if isinstance(layer, keras.layers.GroupNormalization):
                return quantize_annotate_layer(layer, GroupNormQuantizeConfig())
            return layer

        annotated_model = quantize_annotate_model(
            keras.models.clone_model(
                lifted_2d_model, clone_function=apply_quantization_to_gn
            ),
        )

        with quantize_scope({"GroupNormQuantizeConfig": GroupNormQuantizeConfig}):
            q_aware_model = quantize_apply(annotated_model)

        self.model = q_aware_model

        # use legacy adam optimizer for macOS
        if platform.system() == "Darwin":
            optimizer = keras.optimizers.legacy.Adam(self.fine_tune_lr)
        else:
            optimizer = keras.optimizers.Adam(self.fine_tune_lr)

        self.model.compile(
            optimizer=optimizer,
            loss=loss_factory(
                loss_fn_names=["masked_mse"],
                loss_params={},
                loss_weights=[tf.Variable(1.0)],
            ),
            metrics=MaskedCorrelationMetric(),
        )

    def train(self):
        if not self.config.full_config_path.exists():
            self.config.materialize()

        original_batch_size = self.training_dataset_config.batch_size

        self.prepare_dataset()
        self.prepare_model()

        # fine-tune the model
        if isinstance(self.fine_tune_epochs, int):
            self.model.fit(
                self.training_dataset.batch(original_batch_size),
                epochs=self.fine_tune_epochs,
                verbose=1,
            )
        else:
            # Fractional epochs - use steps_per_epoch with repeating dataset
            # 35530 is the number of samples in the training dataset
            total_steps = int(self.fine_tune_epochs * 35530 // original_batch_size)
            self.model.fit(
                self.training_dataset.shuffle(35530).repeat().batch(original_batch_size),
                steps_per_epoch=total_steps,
                epochs=1,
                verbose=1,
            )
