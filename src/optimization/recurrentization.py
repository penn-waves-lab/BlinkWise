import math
import os
from functools import reduce, partial
from itertools import accumulate
from typing import Sequence, Literal

import tensorflow as tf

from src.models.networks.unet import enc_dec_module, input_processing_mlp, output_processing_module, encoder_block_conv
from src.models.config import reverse_nested_sequence, UNetConfig

if os.environ.get("BLINKWISE_USE_TFMOT_KERAS", "0") == "0":
    from tensorflow import keras
else:
    from tensorflow_model_optimization.python.core.keras.compat import keras


# helper function to convert unet level to layer index
def unet_level_to_layer_index(
        level: int,
        encoder_kernel_sizes: Sequence[Sequence[int]],
        pool_sizes: Sequence[int]
) -> int:
    """
    level starts from 1. norm layers are omitted as they have a field-of-view and jump of 1.
    """
    return reduce(lambda r, l: r + len(l), encoder_kernel_sizes[:level], 0) + len(pool_sizes[:level])


def parse_model_config_to_operator_list(
        encoder_kernel_sizes: Sequence[Sequence[int]],
        decoder_kernel_sizes: Sequence[Sequence[int]],
        pool_sizes: Sequence[int],
        final_layer_index: int = -1,
) -> tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """
    Parse the model configuration to get the receptive field, jump, and effective padding for the final layer.

    Args:
        encoder_kernel_sizes: A sequence of sequences where each inner sequence contains the kernel sizes of the encoder.
        decoder_kernel_sizes: A sequence of sequences where each inner sequence contains the kernel sizes of the decoder.
        pool_sizes: A sequence of pool sizes for the encoder and decoder.
        final_layer_index: The index of the final layer. Default is -1, which means the output layer.

    Returns:
        tuple[Sequence[int], Sequence[int], Sequence[int]]: A tuple containing the receptive field, jump, and effective padding for the final layer.
    """
    if final_layer_index == 0:
        raise ValueError("Layer starts from 1. 0 represents input. Negative index is supported.")
    elif final_layer_index > 0:
        # later indexing uses 0-based index. thus we minus 1.
        final_layer_index -= 1
    else:
        # negative indexing is not influenced.
        pass

    upsampling_sizes = list(map(lambda p: 1 / p, reversed(pool_sizes)))

    # flatten the structure to a single list of layers with kernel size and stride
    layer_parameters = []
    for block, pool in zip(encoder_kernel_sizes, pool_sizes):
        for kernel in block:
            layer_parameters.append((kernel, 1, math.ceil((kernel - 1) / 2)))
        layer_parameters.append((pool, pool, 0))
    for block, pool in zip(decoder_kernel_sizes, upsampling_sizes):
        layer_parameters.append((pool, pool, 0))
        for kernel in block:
            layer_parameters.append((kernel, 1, math.ceil((kernel - 1) / 2)))

    # calculate receptive fields, jumps, effective pad
    rf_map, jump_map, effective_pad_map = calculate_rf_and_jump(layer_parameters)
    return rf_map[final_layer_index], jump_map[final_layer_index], effective_pad_map[final_layer_index]


def calculate_rf_and_jump(
        layer_parameters: Sequence[tuple[int, int, int]],
) -> tuple[Sequence[Sequence[int]], Sequence[Sequence[int]], Sequence[Sequence[int]]]:
    """
    Calculate the receptive field and jump for each layer's output with respect to all previous layers' outputs.

    Layers are numbered from l = 1 to L, where L is the total number of layers. The output from each layer l is
    denoted as f_l, with the input as f_0. The l-th layer has a kernel size of k_l, a stride of s_l, and a padding on
    one side of p_l. Starting from f_l, the receptive field (rf_{l, l'}), jump (j_{l, l'}), and effective padding p_{
    l, l') over the output from all previous layers l' = 0, 1, ..., l are calculated. Trivially, rf_{l, l} = j_{l,
    l} = 1, and p_{l, l} = 0.

    The calculations are done recursively as follows:
        rf_{l, l' - 1} = s_l * rf_{l, l'} + (k_l - s_l)       (1)
        j_{l, l' - 1} = s_l * j_{l, l'}                       (2)
        p_{l, l' = 1} = s_l * p_{l, l'} + p_l                 (3)

    Args:
        layer_parameters (Sequence[tuple[int, int, int]]): A sequence of tuples where each tuple contains the kernel
        size, stride and padding of a layer.

    Returns:
        tuple[Sequence[Sequence[int]], Sequence[Sequence[int]], Sequence[Sequence[int]]]: A tuple containing the
        receptive fields, jumps and effective paddingG. All are returned as nested sequences. The outer sequence
        denotes the layer being used as the final layer, and the inner sequence represents the receptive
        field/jump/effective padding of its previous layers up to the input.

    See also:
        Araujo, André, Wade Norris, and Jack Sim. "Computing receptive fields of convolutional neural networks."
        Distill 4.11 (2019): e21. https://distill.pub/2019/computing-receptive-fields/
    """
    n = len(layer_parameters)

    # calculate receptive fields and jumps from each layer output to each layer's input
    rf_map = []
    jump_map = []
    effective_pad_map = []
    #   starting from the first layer to the last layer.
    for l in range(1, n + 1):
        # initialize rf/j/pad_{l, l}. In this case, l' = l.
        rf_list = [1]
        jump_list = [1]
        effective_pad_list = [0]
        current_jump = jump_list[0]
        current_rf = rf_list[0]
        current_effective_pad = effective_pad_list[0]
        # start iteration. l' = l - 1, ..., 0.
        for l_prime in range(l - 1, -1, -1):
            kernel, stride, pad = layer_parameters[l_prime]
            # taking ceil because kernel size/stride can be fractional number
            current_rf = math.ceil(current_rf * stride + (kernel - stride))
            current_jump = math.ceil(current_jump * stride)
            current_effective_pad = math.ceil(current_effective_pad * stride + pad)
            rf_list.append(current_rf)
            jump_list.append(current_jump)
            effective_pad_list.append(current_effective_pad)
        #   reverse the order to start from the input layer to itself.
        rf_map.append(rf_list[::-1])
        jump_map.append(jump_list[::-1])
        effective_pad_map.append(effective_pad_list[::-1])

    return rf_map, jump_map, effective_pad_map


def recurrentized_enc_module(
        convert_to_level: int,
        input_processing_hidden_dims: Sequence[int] = (64, 128, 64),
        encoder_hidden_dims: Sequence[Sequence[int]] = ((64,), (64,), (128,)),
        encoder_kernel_sizes: Sequence[Sequence[int]] = ((7,), (5,), (3,)),
        decoder_hidden_dims: Sequence[Sequence[int]] = ((128,), (64,), (64,)),
        pool_sizes: Sequence[int] = (2, 2, 2),
        encoder_conv_type: Literal["1d", "2d"] = "1d",
        activation_type: str = "silu",
        normalization_type: str = "batch",
        enable_separable: bool = False,
        process_skip_connection: bool = False,
) -> Sequence[keras.models.Model]:
    """
    Construct recurrentized encoder blocks up to the specified level. Level 0 means no recurrentization.

    Returns:
        Sequence[keras.models.Model]: A sequence of keras models representing the recurrentized encoder blocks.
    """
    # check if the convert_to_level is valid
    n_levels = len(encoder_hidden_dims)
    if convert_to_level > n_levels:
        raise ValueError(f"convert_to_level should be less than the number of levels in the U-Net. "
                         f"Got {convert_to_level} > {n_levels}.")
    if convert_to_level == 0:
        return []

    all_expected_skip_connection_n_filters = list(reversed(
        [encoder_hidden_dims[-1][-1]] + [d[-1] for d in decoder_hidden_dims[:-1]]
    ))

    # calculate receptive fields, jumps, and needed paddings
    # again, norm layers are omitted.
    _unet_level_to_layer_index = partial(unet_level_to_layer_index, encoder_kernel_sizes=encoder_kernel_sizes,
                                         pool_sizes=pool_sizes)

    final_layer_index = _unet_level_to_layer_index(convert_to_level)
    rf_map, j_map, pad_map = parse_model_config_to_operator_list(
        encoder_kernel_sizes,
        reverse_nested_sequence(encoder_kernel_sizes),
        pool_sizes,
        final_layer_index=final_layer_index,
    )
    # breakpoints for each unet level. 0 is included as the first element.
    level_breakpoints = [_unet_level_to_layer_index(l) for l in range(convert_to_level + 1)]

    # construct models
    level_models = []
    for l in range(convert_to_level):
        level_kernel_sizes = encoder_kernel_sizes[l]

        if l == 0:
            level_input_size = j_map[0] + level_kernel_sizes[0] - 1
            level_input = keras.layers.Input(
                shape=(level_input_size, input_processing_hidden_dims[-1]) if encoder_conv_type == "1d" else (
                    level_input_size, 1, input_processing_hidden_dims[-1]),
                batch_size=1,
                name=f"level_{0}_input",
            )
        else:
            level_breakpoint = level_breakpoints[l]
            level_input_size = rf_map[level_breakpoint]
            level_input = keras.layers.Input(
                shape=(level_input_size, encoder_hidden_dims[l - 1][-1]) if encoder_conv_type == "1d" else (
                    level_input_size, 1, encoder_hidden_dims[l - 1][-1]),
                batch_size=1,
                name=f"level_{l}_input",
            )

        level_output, skip_connection = encoder_block_conv(
            inputs=level_input,
            block_level=l,
            filters_list=encoder_hidden_dims[l],
            kernel_sizes_list=encoder_kernel_sizes[l],
            pool_size=pool_sizes[l],
            conv_type=encoder_conv_type,
            activation_type=activation_type,
            normalization_type=normalization_type,
            padding_type="valid",
            enable_separable=enable_separable,
            process_skip_connection=process_skip_connection,
            expected_skip_connection_n_filters=all_expected_skip_connection_n_filters[l]
        )
        level_model = keras.Model(
            inputs=level_input,
            outputs={"output": level_output, "skip_connection": skip_connection},
            name=f"encoder_level_{l}",
        )
        level_models.append(level_model)

    return level_models


def get_rec_nonrec_models(
        no_batch_input_shape,
        model_config: UNetConfig,
        convert_to_level: int = 2,
        lift_to_2d: bool = False,
        fix_batch_size: bool = False
) -> tuple[keras.models.Model, Sequence[keras.models.Model], keras.models.Model]:
    """
    returns recurrenrized and non-recurrentized parts of the entire model.

    Args:
        no_batch_input_shape: Shape of the input radar data without the batch size dimension.
        model_config: Model configuration.
        convert_to_level: The level up to which recurrentization is applied. Level 0 means no recurrentization.
        lift_to_2d: whether to lift all 1D operations to 2D.
        fix_batch_size: whether to fix the input batch size to 1.
    """
    # ------------------------------
    # recurrentized part of the model
    # ------------------------------
    # input processing module. after recurrentization the input shape is (batch_size, 1, n_channels), i.e., per frame.
    uniform_length, range_bins, n_channels = no_batch_input_shape

    if fix_batch_size:
        recurrentized_per_frame_input = tf.keras.Input(shape=(1, range_bins, n_channels), batch_size=1)
    else:
        recurrentized_per_frame_input = tf.keras.Input(shape=(1, range_bins, n_channels))

    processed_inputs = input_processing_mlp(
        recurrentized_per_frame_input,
        hidden_dims=model_config.input_processing_hidden_dims,
        activation_type=model_config.input_processing_activation_type,
        lift_to_2d=lift_to_2d
    )
    input_processing_module = keras.models.Model(
        recurrentized_per_frame_input,
        processed_inputs,
        name="input_processing_module"
    )

    # recurrentized encoder blocks
    level_models = recurrentized_enc_module(
        convert_to_level=convert_to_level,
        input_processing_hidden_dims=model_config.input_processing_hidden_dims,
        encoder_hidden_dims=model_config.encoder_hidden_dims,
        encoder_kernel_sizes=model_config.encoder_kernel_sizes,
        decoder_hidden_dims=model_config.decoder_hidden_dims,
        pool_sizes=model_config.encoder_decoder_pool_sizes,
        encoder_conv_type="2d" if lift_to_2d else "1d",
        activation_type=model_config.activation_type,
        normalization_type=model_config.normalization_type,
        enable_separable=model_config.enable_separable,
        process_skip_connection=model_config.process_skip_connection
    )

    # ------------------------------
    # non-recurrentized part of the model
    # ------------------------------
    down_sampling_rate = list(accumulate([1] + list(model_config.encoder_decoder_pool_sizes), lambda x, y: x * y))
    skip_features_lengths = [
        int(uniform_length / r) for r in down_sampling_rate[:convert_to_level]
    ]

    remaining_model_input_shape = (
        int(uniform_length / down_sampling_rate[convert_to_level]),
        1,
        level_models[-1].output["output"].shape[-1]
    ) if lift_to_2d else (
        int(uniform_length / down_sampling_rate[convert_to_level]),
        level_models[-1].output["output"].shape[-1]
    )

    skip_connection_input_shapes = [
        (
            skip_features_lengths[l],
            1,
            level_models[l].output["skip_connection"].shape[-1]
        ) if lift_to_2d else (
            skip_features_lengths[l],
            level_models[l].output["skip_connection"].shape[-1]
        ) for l in range(len(level_models))
    ]

    if fix_batch_size:
        remaining_model_input = tf.keras.Input(shape=remaining_model_input_shape, name="remaining_input", batch_size=1)
        skip_connection_inputs = [tf.keras.Input(shape=shape, name=f"skip_connection_input_{i}", batch_size=1) for
                                  i, shape in enumerate(skip_connection_input_shapes)]
    else:
        remaining_model_input = tf.keras.Input(shape=remaining_model_input_shape, name="remaining_input")
        skip_connection_inputs = [tf.keras.Input(shape=shape, name=f"skip_connection_input_{i}") for i, shape in
                                  enumerate(skip_connection_input_shapes)]

    remaining_part = enc_dec_module(
        remaining_model_input,
        skip_connection_inputs,
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
        remaining_part,
        output_activation=model_config.output_activation,
        conv_type="2d" if lift_to_2d else "1d",
    )

    remaining_model_input_dict = {
        "remaining_input": remaining_model_input,
    }
    for i in range(len(level_models)):
        remaining_model_input_dict[f"skip_connection_input_{i}"] = skip_connection_inputs[i]

    non_recurrentized_model = keras.models.Model(
        inputs=remaining_model_input_dict,
        outputs=outputs,
        name="remaining_model"
    )
    return input_processing_module, level_models, non_recurrentized_model


def get_patch_to_patch_model(
        no_batch_input_shape,
        model_config: UNetConfig,
        convert_to_level: int = 2,
        lift_to_2d: bool = False,
        fix_batch_size: bool = False
) -> keras.models.Model:
    # compute the receptive field of each output element with respect to the input
    final_layer_index = unet_level_to_layer_index(convert_to_level, model_config.encoder_kernel_sizes,
                                                  model_config.encoder_decoder_pool_sizes)
    rf_map, j_map, pad_map = parse_model_config_to_operator_list(
        model_config.encoder_kernel_sizes,
        reverse_nested_sequence(model_config.encoder_kernel_sizes),
        model_config.encoder_decoder_pool_sizes,
        final_layer_index=final_layer_index,
    )

    uniform_length, range_bins, n_channels = no_batch_input_shape

    p2p_input_shape = (rf_map[0], range_bins, n_channels)
    if fix_batch_size:
        p2p_input = tf.keras.Input(shape=p2p_input_shape, name="p2p_input", batch_size=1)
    else:
        p2p_input = tf.keras.Input(shape=p2p_input_shape, name="p2p_input")
    # input processing model
    processed_inputs = input_processing_mlp(
        p2p_input,
        hidden_dims=model_config.input_processing_hidden_dims,
        activation_type=model_config.input_processing_activation_type,
        lift_to_2d=lift_to_2d
    )

    # p2p converted model (up to convert_to_level).
    all_expected_skip_connection_n_filters = list(reversed(
        [model_config.encoder_hidden_dims[-1][-1]] + [d[-1] for d in model_config.decoder_hidden_dims[:-1]]
    ))
    skip_connections = []
    for l in range(convert_to_level):
        processed_inputs, skip_connection = encoder_block_conv(
            inputs=processed_inputs,
            block_level=l,
            filters_list=model_config.encoder_hidden_dims[l],
            kernel_sizes_list=model_config.encoder_kernel_sizes[l],
            pool_size=model_config.encoder_decoder_pool_sizes[l],
            conv_type="2d" if lift_to_2d else "1d",
            activation_type=model_config.activation_type,
            normalization_type=model_config.normalization_type,
            padding_type="valid",
            enable_separable=model_config.enable_separable,
            process_skip_connection=model_config.process_skip_connection,
            expected_skip_connection_n_filters=all_expected_skip_connection_n_filters[l]
        )
        skip_connections.append(skip_connection)

    output_dict = {"output": processed_inputs}
    for i, skip_connection in enumerate(skip_connections):
        output_dict[f"skip_connection_{i}"] = skip_connection

    p2p_model = keras.models.Model(
        p2p_input,
        output_dict,
        name="p2p_model"
    )
    return p2p_model
