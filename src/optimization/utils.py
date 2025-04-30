from typing import Literal

import numpy as np
from src.core import validate_literal_args


@validate_literal_args
def calculate_layer_memory(layer, unit: Literal["float32", "int8"] = "float32"):
    """
    Calculate memory requirements for a layer.

    Args:
        layer: Keras layer
        unit: Memory unit, either 'float32' (4 bytes) or 'int8' (1 byte)

    Returns:
        Memory size in bytes
    """
    # Set bytes per element based on unit type
    if unit == "float32":
        bytes_per_element = 4
    elif unit == "int8":
        bytes_per_element = 1
    else:
        raise ValueError("unit must be either 'float32' or 'int8'")

    # Handle layers with multiple inputs
    if isinstance(layer.input_shape, list):
        input_shapes = layer.input_shape
        input_size = sum(
            np.prod(shape[1:]) for shape in input_shapes if shape is not None
        )
    else:
        # For layers with single input
        input_size = np.prod(layer.input_shape[1:])

    # Handle layers with multiple outputs
    if isinstance(layer.output_shape, list):
        output_shapes = layer.output_shape
        output_size = sum(
            np.prod(shape[1:]) for shape in output_shapes if shape is not None
        )
    else:
        # For layers with single output
        output_size = np.prod(layer.output_shape[1:])

    # Sum input and output memory with specified unit size
    return (input_size + output_size) * bytes_per_element


def is_include_layer(layer, deny_layer_type_list: list[str]):
    return all([deny not in layer.name for deny in deny_layer_type_list])
