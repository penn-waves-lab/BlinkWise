import os

import numpy as np
import tensorflow_model_optimization as tfmot

if os.environ.get("BLINKWISE_USE_TFMOT_KERAS", "0") == "0":
    from tensorflow import keras
else:
    from tensorflow_model_optimization.python.core.keras.compat import keras

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer


class GroupNormQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """Quantize config for GroupNormalization layer (weights only)"""

    def __init__(self):
        self.weight_quantizer = LastValueQuantizer(
            num_bits=8, symmetric=True, narrow_range=False, per_axis=False
        )

    def get_weights_and_quantizers(self, layer):
        return [
            (layer.gamma, self.weight_quantizer),
            (layer.beta, self.weight_quantizer),
        ]

    def set_quantize_weights(self, layer, quantize_weights):
        layer.gamma = quantize_weights[0]
        layer.beta = quantize_weights[1]

    def get_activations_and_quantizers(self, layer):
        # GroupNorm doesn't have activation
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        # No activations to set
        pass

    def get_output_quantizers(self, layer):
        # Let TFLite handle output quantization
        return []

    def get_config(self):
        return {}


class ParameterTransfer1Dto2D:
    """Transfer parameters from 1D model to 2D model with matching layer names."""

    @staticmethod
    def transfer_parameters(
            model_1d: keras.models.Model, model_2d: keras.models.Model
    ) -> dict:
        """
        Transfer parameters from 1D to 2D model.

        Args:
            model_1d: Source 1D model
            model_2d: Target 2D model

        Returns:
            Dict containing transfer statistics
        """
        stats = {"matched_layers": 0, "transferred_layers": 0, "skipped_layers": []}

        # Get layers by name
        layers_1d = {layer.name: layer for layer in model_1d.layers}
        layers_2d = {layer.name: layer for layer in model_2d.layers}

        # Find matching layers and transfer parameters
        for name, layer_2d in layers_2d.items():
            if name in layers_1d:
                stats["matched_layers"] += 1
                layer_1d = layers_1d[name]

                # Skip if layer has no weights
                if not layer_1d.weights:
                    stats["skipped_layers"].append(
                        {"name": name, "reason": "no weights"}
                    )
                    continue

                try:
                    ParameterTransfer1Dto2D._transfer_layer_parameters(
                        layer_1d, layer_2d
                    )
                    stats["transferred_layers"] += 1
                except Exception as e:
                    stats["skipped_layers"].append({"name": name, "reason": str(e)})

        return stats

    @staticmethod
    def _transfer_layer_parameters(
            layer_1d: keras.layers.Layer, layer_2d: keras.layers.Layer
    ) -> None:
        """
        Transfer parameters between corresponding layers.
        """
        weights_1d = layer_1d.get_weights()

        if isinstance(layer_1d, keras.layers.Conv1D) and isinstance(
                layer_2d, keras.layers.Conv2D
        ):
            reshaped_weights = ParameterTransfer1Dto2D._reshape_conv_weights(weights_1d)
            layer_2d.set_weights(reshaped_weights)

        elif isinstance(layer_1d, keras.layers.SeparableConv1D) and isinstance(
                layer_2d, keras.layers.SeparableConv2D
        ):
            reshaped_weights = ParameterTransfer1Dto2D._reshape_separable_conv_weights(
                weights_1d
            )
            layer_2d.set_weights(reshaped_weights)

        elif isinstance(layer_1d, keras.layers.DepthwiseConv1D) and isinstance(layer_2d, keras.layers.DepthwiseConv2D):
            # For depthwise conv, weights shape is (kernel_size, input_channels, depth_multiplier)
            # Need to expand kernel_size dimension: (kernel_size, 1, in_channels, depth_multiplier)
            w = layer_1d.get_weights()
            kernel_1d = w[0]
            kernel_2d = np.expand_dims(kernel_1d, axis=1)
            if len(w) > 1:  # If bias exists
                layer_2d.set_weights([kernel_2d, w[1]])
            else:
                layer_2d.set_weights([kernel_2d])

        elif isinstance(layer_1d, keras.layers.GroupNormalization) and isinstance(
                layer_2d, keras.layers.GroupNormalization
        ):
            # GroupNorm parameters can be transferred directly
            layer_2d.set_weights(weights_1d)

        elif isinstance(layer_1d, keras.layers.Dense) and isinstance(
                layer_2d, keras.layers.Dense
        ):
            # Dense layer parameters can be transferred directly
            layer_2d.set_weights(weights_1d)

        else:
            # For any other matching layer types, try direct transfer
            layer_2d.set_weights(weights_1d)

    @staticmethod
    def _reshape_conv_weights(weights_1d: list[np.ndarray]) -> list[np.ndarray]:
        """
        Reshape 1D convolution weights to 2D format.
        """
        reshaped_weights = []

        # Reshape kernel: (kernel_size, in_channels, filters) -> (kernel_size, 1, in_channels, filters)
        kernel = weights_1d[0]
        reshaped_kernel = np.expand_dims(kernel, axis=1)
        reshaped_weights.append(reshaped_kernel)

        # Add bias if present
        if len(weights_1d) > 1:
            reshaped_weights.append(weights_1d[1])

        return reshaped_weights

    @staticmethod
    def _reshape_separable_conv_weights(
            weights_1d: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Reshape 1D separable convolution weights to 2D format.
        """
        reshaped_weights = []

        # Depthwise kernel: (kernel_size, in_channels, depth_multiplier) -> (kernel_size, 1, in_channels, depth_multiplier)
        depthwise_kernel = weights_1d[0]
        reshaped_depthwise = np.expand_dims(depthwise_kernel, axis=1)
        reshaped_weights.append(reshaped_depthwise)

        # Pointwise kernel: (1, in_channels * depth_multiplier, filters) -> (1, 1, in_channels * depth_multiplier, filters)
        pointwise_kernel = weights_1d[1]
        reshaped_pointwise = np.expand_dims(pointwise_kernel, axis=1)
        reshaped_weights.append(reshaped_pointwise)

        # Add bias if present
        if len(weights_1d) > 2:
            reshaped_weights.append(weights_1d[2])

        return reshaped_weights


def transfer_parameters(
        model_1d: keras.models.Model, model_2d: keras.models.Model, verbose: bool = True
) -> dict:
    """
    Transfer parameters from 1D model to 2D model and optionally print transfer statistics.

    Args:
        model_1d: Source 1D model. The one we normally trained.
        model_2d: Target 2D model. The model to be compatible with TFMOT.
        verbose: Whether to print transfer statistics

    Returns:
        Dictionary containing transfer statistics
    """
    transfer_util = ParameterTransfer1Dto2D()
    stats = transfer_util.transfer_parameters(model_1d, model_2d)

    if verbose:
        print(f"Parameter transfer statistics:")
        print(f"- Matched layers: {stats['matched_layers']}")
        print(f"- Successfully transferred: {stats['transferred_layers']}")
        if stats["skipped_layers"]:
            print("\nSkipped layers:")
            for skip in stats["skipped_layers"]:
                print(f"- {skip['name']}: {skip['reason']}")

    return stats
