from typing import get_type_hints, Literal, get_args, Sequence, Union
from functools import wraps
from inspect import signature
from collections.abc import Sequence as SequenceType


def validate_literal_args(func):
    """
    Decorator to validate Literal typed parameters.

    Arguments with Literal type hints will be checked to ensure that the value passed is one of the specified literals.

    Args:
        func: Function to decorate
    """
    hints = get_type_hints(func)

    def _extract_literal_info(type_hint, optional):
        if not hasattr(type_hint, "__origin__"):
            return None

        # Handle Union types (including Optional)
        if type_hint.__origin__ is Union:
            # Try extracting Literal info from each Union member
            for union_type in type_hint.__args__:
                info = _extract_literal_info(union_type, optional=True)
                if info:
                    return info
            return None

        if type_hint.__origin__ is Literal:
            return {
                "values": get_args(type_hint),
                "is_sequence": False,
                "optional": optional
            }
        elif type_hint.__origin__ is Sequence or (
            isinstance(type_hint.__origin__, type) and  # Check if it's a class
            issubclass(type_hint.__origin__, SequenceType)
        ):
            # Check if the sequence contains Literal type
            arg = type_hint.__args__[0]
            if hasattr(arg, "__origin__") and arg.__origin__ is Literal:
                return {
                    "values": get_args(arg),
                    "is_sequence": True,
                    "optional": optional
                }
        return None

    # Extract valid values and type info for Literal typed parameters
    literal_params = {}
    for name, hint in hints.items():
        param_info = _extract_literal_info(hint, optional=False)
        if param_info:
            literal_params[name] = param_info

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not literal_params:
            return func(*args, **kwargs)

        # Get bound arguments
        bound = signature(func).bind(*args, **kwargs)
        bound.apply_defaults()

        # Check the Literal typed parameters
        for param_name, param_info in literal_params.items():
            if param_name not in bound.arguments:
                continue

            value = bound.arguments[param_name]

            # Skip validation if value is None and parameter is optional
            if param_info["optional"] and value is None:
                continue

            valid_values = param_info["values"]

            if param_info["is_sequence"]:
                if not isinstance(value, SequenceType):
                    raise TypeError(
                        f"Parameter {param_name} must be a sequence, got {type(value)}"
                    )
                invalid_values = [v for v in value if v not in valid_values]
                if invalid_values:
                    raise ValueError(
                        f"Invalid values in sequence for {param_name}: {invalid_values}. "
                        f"All values must be one of: {valid_values}"
                    )
            else:
                if value not in valid_values:
                    raise ValueError(
                        f"Invalid value for {param_name}: {value}. "
                        f"Must be one of: {valid_values}"
                    )

        return func(*args, **kwargs)

    return wrapper