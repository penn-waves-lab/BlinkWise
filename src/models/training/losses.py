import os
from typing import Sequence, Callable

import tensorflow as tf
if os.environ.get("BLINKWISE_USE_TFMOT_KERAS", "0") == "0":
    from tensorflow import keras
else:
    from tensorflow_model_optimization.python.core.keras.compat import keras

# ===================
# masked mse loss (for regression)
# ===================

def valley_weight(y, y_min=0.0, alpha=2.0, beta=10.0):
    return 1.0 + alpha * tf.exp(-beta * tf.square(y - y_min))


def masked_mse_loss(y_true, y_pred, original_lengths, use_valley_weight=False, y_min=0.0, alpha=2.0, beta=10.0):
    # Create a mask based on original_lengths
    mask = tf.sequence_mask(original_lengths, maxlen=tf.shape(y_true)[1])
    mask = tf.cast(mask, dtype=y_true.dtype)

    # Apply the mask
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    # Compute squared difference
    squared_difference = tf.square(y_true_masked - y_pred_masked)

    if use_valley_weight:
        # Compute weights based on y_true if valley weighting is enabled
        weights = valley_weight(y_true_masked, y_min, alpha, beta)
        squared_difference = weights * squared_difference

    sum_squared_difference = tf.reduce_sum(squared_difference)

    # Count the number of unmasked values
    count_unmasked = tf.reduce_sum(mask)

    return sum_squared_difference / count_unmasked


class MaskedMSELoss(keras.losses.Loss):
    def __init__(self, use_valley_weight=False, y_min=0.0, alpha=2.0, beta=10.0, name="masked_mse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.use_valley_weight = use_valley_weight
        self.y_min = y_min
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        # y_true: B x UL, y_pred: B x UL. (not UL x 1 here).
        # Assuming original_lengths is passed as part of y_true
        original_lengths = tf.cast(y_true[:, -1], dtype=tf.int32)
        y_true = y_true[:, :-1]  # Remove the last time step containing original_lengths
        return masked_mse_loss(y_true, y_pred, original_lengths,
                               self.use_valley_weight, self.y_min, self.alpha, self.beta)

    def get_config(self):
        config = super().get_config()
        config.update({
            "use_valley_weight": self.use_valley_weight,
            "y_min": self.y_min,
            "alpha": self.alpha,
            "beta": self.beta
        })
        return config

# ===================
# penalty over high frequency components from the model. (for regression)
# ===================

def padded_spectral_loss(y_true, y_pred, original_lengths, cutoff_frequency=0.1, padding_penalty=1.0):
    max_len = tf.shape(y_pred)[1]

    def process_sequence(args):
        sequence, length = args
        # Split into valid and padded regions
        padded_seq = sequence[length:]

        # Spectral loss for valid region
        fft = tf.signal.rfft(sequence)
        magnitude_spectrum = tf.abs(fft)
        freqs = tf.linspace(0.0, 0.5, tf.shape(magnitude_spectrum)[0])
        high_pass_filter = tf.cast(freqs > cutoff_frequency, sequence.dtype)
        high_frequency_content = magnitude_spectrum * high_pass_filter
        spectral_loss = tf.reduce_sum(tf.square(high_frequency_content)) / tf.cast(length, tf.float32)

        # Flatness loss for padded region
        if length < max_len:
            last_valid_value = sequence[length - 1]
            flatness_loss = tf.reduce_mean(tf.square(padded_seq - last_valid_value))
        else:
            flatness_loss = 0.0

        return spectral_loss + padding_penalty * flatness_loss

    # Process each sequence individually
    sequence_losses = tf.map_fn(
        process_sequence,
        (y_pred, original_lengths),
        fn_output_signature=tf.float32
    )

    return tf.reduce_mean(sequence_losses)


class PaddedSpectralLoss(keras.losses.Loss):
    def __init__(self, cutoff_frequency=0.1, padding_penalty=1.0, name="padded_spectral_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cutoff_frequency = cutoff_frequency
        self.padding_penalty = padding_penalty

    def call(self, y_true, y_pred):
        # y_true: B x (UL+1), y_pred: B x UL x 1
        original_lengths = tf.cast(y_true[:, -1], dtype=tf.int32)
        y_true = y_true[:, :-1]  # Remove the last column containing original_lengths
        return padded_spectral_loss(y_true, y_pred, original_lengths, self.cutoff_frequency, self.padding_penalty)

    def get_config(self):
        config = super().get_config()
        config.update({
            "cutoff_frequency": self.cutoff_frequency,
            "padding_penalty": self.padding_penalty
        })
        return config


# ===================
# public interfaces to acquire a combined loss.
# ===================


def loss_factory(
        loss_fn_names: Sequence[str],
        loss_params: dict,
        loss_weights: Sequence[tf.Variable] = None,
):
    """
    A factory function outputs

    see callbacks.WeightScheduler to schedule loss weights during training.
    """
    valid_names = {
        "masked_mse": MaskedMSELoss(
            use_valley_weight=loss_params.get("use_valley_weight", False),
            y_min=loss_params.get("y_min", 0.0),
            alpha=loss_params.get("alpha", 2.0),
            beta=loss_params.get("beta", 10.0),
        ),
        "padded_spectral": PaddedSpectralLoss(
            cutoff_frequency=loss_params.get("cutoff_frequency", 0.1),
            padding_penalty=loss_params.get("padding_penalty", 1.0),
        ),
    }

    # Define the combined loss function
    def combined_loss(y_true, y_pred):
        total_loss = 0.0
        for name, weight in zip(loss_fn_names, loss_weights):
            loss_function = valid_names[name]
            total_loss += keras.backend.get_value(weight) * loss_function(y_true, y_pred)
        return total_loss

    return combined_loss


def loss_fn_names_to_list_of_funcs(
        loss_fn_names: Sequence[str],
        loss_params: dict,
) -> list[Callable]:
    valid_names = {
        "masked_mse": MaskedMSELoss(
            use_valley_weight=loss_params.get("use_valley_weight", False),
            y_min=loss_params.get("y_min", 0.0),
            alpha=loss_params.get("alpha", 2.0),
            beta=loss_params.get("beta", 10.0),
        ),
        "padded_spectral": PaddedSpectralLoss(
            cutoff_frequency=loss_params.get("cutoff_frequency", 0.1),
            padding_penalty=loss_params.get("padding_penalty", 1.0),
        ),
    }

    func_list = []
    for name in loss_fn_names:
        func_list.append(valid_names[name])
    return func_list
