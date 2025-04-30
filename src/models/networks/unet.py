import os
from typing import Sequence, Literal

import tensorflow as tf
from ..config import UNetConfig

if os.environ.get("BLINKWISE_USE_TFMOT_KERAS", "0") == "0":
    from tensorflow import keras
else:
    from tensorflow_model_optimization.python.core.keras.compat import keras

# =========================
# input processing modules
# =========================

def input_processing_mlp(
        inputs,
        hidden_dims=(64, 128, 64),
        activation_type: str = "silu",
        lift_to_2d: bool = False
):
    """
    applicable to input shape B x T x R x A. R x A will be flattened into a feature.
    """
    n_layers = len(hidden_dims)

    x = keras.layers.SeparableConv2D(
        hidden_dims[0],
        (1, inputs.shape[2]),
        padding="valid",
        name=f"input_sep_conv",
    )(inputs)
    x = keras.layers.Activation(activation_type, name=f"input_rep_conv_act")(x)
    x = keras.layers.Reshape((x.shape[1], x.shape[-1]), name="input_flatten_reshape")(x)
    for i in range(n_layers):
        # fully connection layers
        x = keras.layers.Dense(hidden_dims[i], name=f"input_fc_{i}")(x)
        if i == n_layers // 2:
            if lift_to_2d:
                # lift the feature to 2D by adding a dummy dimension
                x = keras.layers.Reshape((x.shape[1], 1, x.shape[-1]), name="input_dummy_reshape_lift")(x)
                x = keras.layers.DepthwiseConv2D((1, 1), name=f"input_fc_depthwise_{i}")(x)
                x = keras.layers.Reshape((x.shape[1], x.shape[-1]), name="input_dummy_reshape_squeeze")(x)
            else:
                x = keras.layers.DepthwiseConv1D(1, name=f"input_fc_depthwise_{i}")(x)
        x = keras.layers.Activation(activation_type, name=f"input_fc_act_{i}")(x)

    if lift_to_2d:
        # lift the feature to 2D by adding a dummy dimension
        x = keras.layers.Reshape((x.shape[1], 1, x.shape[-1]), name="input_reshape_1d_to_2d")(x)

    return x


# =========================
# unet blocks
# =========================
def encoder_block_conv(
        inputs,
        block_level: int,
        filters_list: Sequence[int],
        kernel_sizes_list: Sequence[int],
        pool_size: int,
        conv_type: Literal["1d", "2d"] = "1d",
        activation_type: str = "silu",
        normalization_type: str = "batch",
        padding_type: str = "same",
        enable_separable: bool = False,
        process_skip_connection: bool = False,
        expected_skip_connection_n_filters: int = None
):
    """
    Constructs an encoder block with multiple (separable) convolutional layers, normalization, activation,
    pooling and optional temporal shift for a U-Net like architecture.

    Returns:
        Tuple[Tensor, Tensor]: The downsampled tensor and the skip connection tensor.
    """
    # initialization
    if conv_type == "2d":
        # some of the downstream libraries require 2D operations, such as tfmot.
        # we manually lift 1D operation to 2D by adding a dummy dimension.
        conv_layer = keras.layers.SeparableConv2D if enable_separable else keras.layers.Conv2D
        kernel_sizes_list = map(lambda k: (k, 1), kernel_sizes_list)
        pool_size = (pool_size, 1)
        pooling_layer = keras.layers.MaxPooling2D
    else:
        conv_layer = keras.layers.SeparableConv1D if enable_separable else keras.layers.Conv1D
        pooling_layer = keras.layers.MaxPooling1D

    # multiple conv layers at one level
    x = inputs
    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_sizes_list)):
        x = conv_layer(filters, kernel_size, padding=padding_type, name=f"enc_conv_{block_level}_{i}")(x)

        norm_layer_name = f"enc_norm_{block_level}_{i}"
        if normalization_type == "instance":
            x = keras.layers.GroupNormalization(groups=-1, name=norm_layer_name)(x)
        else:  # group normalization
            x = keras.layers.BatchNormalization(name=norm_layer_name)(x)

        x = keras.layers.Activation(activation_type, name=f"enc_act_{block_level}_{i}")(x)

    # manage skip connections
    skip_features = x  # We take the last output as skip connection
    if process_skip_connection:
        if expected_skip_connection_n_filters is None:
            raise ValueError("Expected_skip_connection_n_filters must be provided when process_skip_connection is set.")

        skip_features = conv_layer(
            expected_skip_connection_n_filters,
            1 if conv_type == "1d" else (1, 1),
            1,
            name=f"enc_skip_conv_{block_level}"
        )(skip_features)

    # downsampling
    x = pooling_layer(pool_size, name=f"enc_pooling_{block_level}")(x)

    return x, skip_features


def decoder_block_conv(
        inputs,
        skip_features,
        block_level: int,
        filters_list: Sequence[int],
        kernel_sizes_list: Sequence[int],
        pool_size: int,
        conv_type: Literal["1d", "2d"] = "1d",
        activation_type: str = "silu",
        normalization_type: str = "batch",
        enable_separable: bool = False,
        process_skip_connection: bool = False,
):
    """
    Constructs a decoder block with multiple (separable) convolutional layers, normalization, activation, upsampling
    for a U-Net like architecture.

    Returns:
        Tensor: The upsampled tensor after concatenation and convolutional processing.
    """
    # initialization
    if conv_type == "2d":
        # some of the downstream libraries require 2D operations, such as tfmot.
        # we manually lift 1D operation to 2D by adding a dummy dimension.
        conv_layer = keras.layers.SeparableConv2D if enable_separable else keras.layers.Conv2D
        kernel_sizes_list = map(lambda k: (k, 1), kernel_sizes_list)
        pool_size = (pool_size, 1)
        upsampling_layer = keras.layers.UpSampling2D
    else:
        conv_layer = keras.layers.SeparableConv1D if enable_separable else keras.layers.Conv1D
        upsampling_layer = keras.layers.UpSampling1D

    x = inputs
    x = upsampling_layer(pool_size, name=f"dec_upsampling_{block_level}")(x)

    if process_skip_connection:
        x = keras.layers.Add(name=f"dec_sc_add_{block_level}")([x, skip_features])
    else:
        x = keras.layers.Concatenate(name=f"dec_sc_concat_{block_level}")([x, skip_features])

    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_sizes_list)):
        x = conv_layer(filters, kernel_size, padding='same', name=f"dec_conv_{block_level}_{i}")(x)

        norm_layer_name = f"dec_norm_{block_level}_{i}"
        if normalization_type == "instance":
            x = keras.layers.GroupNormalization(groups=-1, name=norm_layer_name)(x)
        else:  # group normalization
            x = keras.layers.BatchNormalization(name=norm_layer_name)(x)

        x = keras.layers.Activation(activation_type, name=f"dec_act_{block_level}_{i}")(x)

    return x


def bottleneck_module(
        inputs,
        bottleneck_dim,
        conv_type: Literal["1d", "2d"] = "1d",
        activation_type: str = "silu",
        normalization_type: str = "batch",
        enable_separable: bool = False
):
    if conv_type == "2d":
        conv_func = keras.layers.SeparableConv2D if enable_separable else keras.layers.Conv2D
    else:
        conv_func = keras.layers.SeparableConv1D if enable_separable else keras.layers.Conv1D

    x = inputs

    # Bottleneck (Using last dimension size of the last encoder layer)
    x = conv_func(bottleneck_dim, 3 if conv_type == "1d" else (3, 1), padding='same', name="bottleneck_conv")(x)

    norm_layer_name = "bottleneck_norm"
    if normalization_type == "instance":
        x = keras.layers.GroupNormalization(groups=-1, name=norm_layer_name)(x)
    else:  # group normalization
        x = keras.layers.BatchNormalization(name=norm_layer_name)(x)

    x = keras.layers.Activation(activation_type, name="bottleneck_act")(x)
    return x


# =========================
# assembled unet
# =========================

def enc_dec_module(
        inputs,
        skip_connections=(),
        encoder_hidden_dims: Sequence[Sequence[int]] = ((64,), (64,), (128,)),
        encoder_kernel_sizes: Sequence[Sequence[int]] = ((7,), (5,), (3,)),
        decoder_hidden_dims: Sequence[Sequence[int]] = ((128,), (64,), (64,)),
        decoder_kernel_sizes: Sequence[Sequence[int]] = ((3,), (5,), (7,)),
        pool_sizes: Sequence[int] = (2, 2, 2),
        activation_type: str = "silu",
        normalization_type: str = "batch",
        enable_separable: bool = False,
        encoder_conv_type: Literal["1d", "2d"] = "1d",
        decoder_conv_type: Literal["1d", "2d"] = "1d",
        process_skip_connection: bool = False
):
    """
    Constructs a U-Net like network with specified convolutions in the encoder and the decoder.

    Returns:
        Tensor: The output tensor of the U-Net like network.
    """
    all_expected_skip_connection_n_filters = list(reversed(
        [encoder_hidden_dims[-1][-1]] + [d[-1] for d in decoder_hidden_dims[:-1]]
    ))

    n_layers = len(encoder_hidden_dims)
    x = inputs

    processed_layers = len(skip_connections)
    # Encoders
    for l in range(processed_layers, n_layers):
        x, skip_connection = encoder_block_conv(
            inputs=x,
            block_level=l,
            filters_list=encoder_hidden_dims[l],
            kernel_sizes_list=encoder_kernel_sizes[l],
            pool_size=pool_sizes[l],
            conv_type=encoder_conv_type,
            activation_type=activation_type,
            normalization_type=normalization_type,
            enable_separable=enable_separable,
            process_skip_connection=process_skip_connection,
            expected_skip_connection_n_filters=all_expected_skip_connection_n_filters[l]
        )
        skip_connections.append(skip_connection)

    # Bottleneck (Using last dimension size of the last encoder layer)
    bottleneck_size = encoder_hidden_dims[-1][-1]
    x = bottleneck_module(
        inputs=x,
        bottleneck_dim=bottleneck_size,
        conv_type=encoder_conv_type,
        activation_type=activation_type,
        normalization_type=normalization_type,
        enable_separable=enable_separable
    )

    # Decoder
    for l in range(len(decoder_hidden_dims)):
        x = decoder_block_conv(
            inputs=x,
            block_level=l,
            skip_features=skip_connections[-(l + 1)],
            filters_list=decoder_hidden_dims[l],
            kernel_sizes_list=decoder_kernel_sizes[l],
            pool_size=pool_sizes[-(l + 1)],
            conv_type=decoder_conv_type,
            activation_type=activation_type,
            normalization_type=normalization_type,
            enable_separable=enable_separable,
            process_skip_connection=process_skip_connection
        )

    return x


# =========================
# output processing modules
# =========================

def output_processing_module(inputs, output_activation="sigmoid", conv_type="1d"):
    """
    a combination of 1d pointwise convolution and softmax
    """
    if conv_type == "2d":
        conv_layer = keras.layers.Conv2D
    else:
        conv_layer = keras.layers.Conv1D
    outputs = conv_layer(
        filters=1,
        kernel_size=1 if conv_type == "1d" else (1, 1),
        activation=output_activation,
        name="score"
    )(inputs)

    if conv_type == "2d":
        # remove the dummy dimension
        outputs = keras.layers.Reshape((outputs.shape[1], 1), name="output_reshape_2d_to_1d")(outputs)

    return outputs

# =========================
# main interfaces
# =========================

def unet_factory(
        no_batch_input_shape,
        model_config: UNetConfig,
        lift_to_2d: bool = False,
        fix_batch_size: bool = False
) -> keras.models.Model:
    if fix_batch_size:
        inputs = tf.keras.Input(shape=no_batch_input_shape, batch_size=1)
    else:
        inputs = tf.keras.Input(shape=no_batch_input_shape)
    # input processing module.
    processed_inputs = input_processing_mlp(
        inputs,
        hidden_dims=model_config.input_processing_hidden_dims,
        activation_type=model_config.input_processing_activation_type,
        lift_to_2d=lift_to_2d
    )
    # encoder-decoder model.
    #   (B, T, C) -> (B, T, C'). 1d conv enc -> 1d conv dec.
    encoded_decoded = enc_dec_module(
        processed_inputs,
        skip_connections=[],
        encoder_hidden_dims=model_config.encoder_hidden_dims,
        encoder_kernel_sizes=model_config.encoder_kernel_sizes,
        decoder_hidden_dims=model_config.decoder_hidden_dims,
        decoder_kernel_sizes=model_config.decoder_kernel_sizes,
        pool_sizes=model_config.encoder_decoder_pool_sizes,
        activation_type=model_config.activation_type,
        normalization_type=model_config.normalization_type,
        enable_separable=model_config.enable_separable,
        encoder_conv_type="2d" if lift_to_2d else "1d",
        decoder_conv_type="2d" if lift_to_2d else "1d",
        process_skip_connection=model_config.process_skip_connection
    )
    # output processing module.
    outputs = output_processing_module(
        encoded_decoded,
        output_activation=model_config.output_activation,
        conv_type="2d" if lift_to_2d else "1d"
    )
    return keras.models.Model(inputs, outputs, name=f"unet1d_segmenter")
