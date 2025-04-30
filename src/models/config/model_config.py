from typing import Literal, Sequence

from ..config.base_config import BaseConfig, reverse_nested_sequence
from src.core import validate_literal_args


class BaseModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()


class UNetConfig(BaseModelConfig):
    n_antennas = 3

    @validate_literal_args
    def __init__(
            self,
            input_processing_type: Literal["mlp"] = "mlp",
            output_processing_type: Literal["direct"] = "direct",
            input_processing_hidden_dims=(32, 64),
            input_processing_activation_type: str = "silu",
            enable_separable: bool = False,
            encoder_hidden_dims: Sequence[Sequence[int]] = ((64,), (64,), (128,)),
            decoder_hidden_dims: Sequence[Sequence[int]] = None,
            encoder_kernel_sizes: Sequence[Sequence[int]] = ((7,), (5,), (3,)),
            decoder_kernel_sizes: Sequence[Sequence[int]] = None,
            encoder_decoder_pool_sizes: Sequence[int] = (4, 2, 2),
            process_skip_connection: bool = False,
            activation_type: str = "silu",
            normalization_type: Literal["batch", "instance"] = "batch",
            output_activation: Literal["sigmoid"] = "sigmoid",
    ):
        """
        Configuration for a U-Net model.

        Args:
            input_processing_type: Type of input processing module. Default is "mlp".
            output_processing_type: Type of output processing module. Default is "direct".
            input_processing_hidden_dims: Hidden dimensions for the input processing module. Default is (32, 64).
            input_processing_activation_type: Activation type for the input processing module. Default is "silu".
            enable_separable: Whether to use separable convolutions. Default is False.
            encoder_hidden_dims: Hidden dimensions for the encoder. Default is ((64,), (64,), (128,)).
            decoder_hidden_dims: Hidden dimensions for the decoder. Default is None. If None, the decoder configuration is the reverse of the encoder configuration.
            encoder_kernel_sizes: Kernel sizes for the encoder. Default is ((7,), (5,), (3,)).
            decoder_kernel_sizes: Kernel sizes for the decoder. Default is None. If None, the decoder configuration is the reverse of the encoder configuration.
            encoder_decoder_pool_sizes: Pool sizes for the encoder and decoder. Default is (4, 2, 2).
            process_skip_connection: Whether to process skip connections. Default is False.
            activation_type: Activation type for the model. Default is "silu".
            normalization_type: Normalization type for the model. Default is "batch".
            output_activation: Activation type for the output. Default is "sigmoid".
        """
        super().__init__()

        self.input_processing_type = input_processing_type
        self.input_processing_activation_type = input_processing_activation_type

        self.output_processing_type = output_processing_type

        # parameters shared by the whole model
        self.enable_separable = enable_separable
        self.activation_type = activation_type
        self.normalization_type = normalization_type

        # input processing module
        self.input_processing_hidden_dims = input_processing_hidden_dims

        # unet params
        #   default for decoder configurations is the reverse of those for encoders
        if decoder_hidden_dims is None:
            decoder_hidden_dims = reverse_nested_sequence(encoder_hidden_dims)
        if decoder_kernel_sizes is None:
            decoder_kernel_sizes = reverse_nested_sequence(encoder_kernel_sizes)
        #   valid configurations provided
        #       1). number of levels should match.
        n_levels = len(encoder_decoder_pool_sizes)
        if any(map(
                lambda x: not len(x) == n_levels,
                [encoder_hidden_dims, decoder_hidden_dims, encoder_kernel_sizes, decoder_kernel_sizes]
        )):
            raise ValueError(f"A {n_levels}-level U-Net is defined, while other network configs do not match.")

        #       2). at each level, the number of hidden dims specified should match that of kernel sizes.
        def _check_per_level_config(hidden_dims, kernel_sizes):
            return all(map(
                lambda hidden_dims_level, kernel_sizes_level: len(hidden_dims_level) == len(kernel_sizes_level),
                hidden_dims,
                kernel_sizes
            ))

        if not _check_per_level_config(encoder_hidden_dims, encoder_kernel_sizes):
            raise ValueError(
                f"number of hidden dimensions and kernel sizes do not match in per level encoder config.")
        if not _check_per_level_config(decoder_hidden_dims, decoder_kernel_sizes):
            raise ValueError(
                f"number of hidden dimensions and kernel sizes do not match in per level decoder config.")

        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.encoder_kernel_sizes = encoder_kernel_sizes
        self.decoder_kernel_sizes = decoder_kernel_sizes
        self.encoder_decoder_pool_sizes = encoder_decoder_pool_sizes

        self.process_skip_connection = process_skip_connection

        # output processing module params
        self.output_activation = output_activation
