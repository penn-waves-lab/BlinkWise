import numpy as np

def convert_history_to_python_types(history):
    """
    Convert all entries in a Keras history object to native Python types.

    Args:
    history (dict): A dictionary containing training/validation loss and metrics.

    Returns:
    dict: A dictionary with all values converted to native Python types.
    """

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    converted_history = {key: [convert(value) for value in values] for key, values in history.items()}
    return converted_history
