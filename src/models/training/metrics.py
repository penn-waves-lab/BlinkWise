import os

import tensorflow as tf

if os.environ.get("BLINKWISE_USE_TFMOT_KERAS", "0") == "0":
    from tensorflow import keras
else:
    from tensorflow_model_optimization.python.core.keras.compat import keras


# ===================
# masked Pearson correlation
# ===================

def masked_correlation(y_true, y_pred, original_lengths):
    # Create a mask based on original_lengths
    mask = tf.sequence_mask(original_lengths, maxlen=tf.shape(y_true)[1])
    mask = tf.cast(mask, dtype=y_true.dtype)

    # Apply the mask
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    # Calculate means
    y_true_mean = tf.reduce_sum(y_true_masked, axis=1) / tf.reduce_sum(mask, axis=1)
    y_pred_mean = tf.reduce_sum(y_pred_masked, axis=1) / tf.reduce_sum(mask, axis=1)

    # Calculate centered masked values
    y_true_centered = (y_true_masked - y_true_mean[:, tf.newaxis]) * mask
    y_pred_centered = (y_pred_masked - y_pred_mean[:, tf.newaxis]) * mask

    # Calculate numerator (covariance)
    numerator = tf.reduce_sum(y_true_centered * y_pred_centered, axis=1)

    # Calculate denominator (standard deviations)
    y_true_std = tf.sqrt(tf.reduce_sum(tf.square(y_true_centered), axis=1))
    y_pred_std = tf.sqrt(tf.reduce_sum(tf.square(y_pred_centered), axis=1))
    denominator = y_true_std * y_pred_std

    # Calculate correlation
    correlation = numerator / (denominator + tf.keras.backend.epsilon())

    # Average correlation across the batch
    return tf.reduce_mean(correlation)


class MaskedCorrelationMetric(keras.metrics.Metric):
    def __init__(self, name='masked_correlation', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_correlation = self.add_weight(name='total_correlation', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: B x UL, y_pred: B x UL x 1. (different from that in loss).
        y_pred = tf.squeeze(y_pred, axis=-1)

        # Assuming original_lengths is passed as part of y_true
        original_lengths = tf.cast(y_true[:, -1], dtype=tf.int32)
        y_true = y_true[:, :-1]  # Remove the last time step containing original_lengths

        correlation = masked_correlation(y_true, y_pred, original_lengths)

        self.total_correlation.assign_add(correlation)
        self.count.assign_add(1)

    def result(self):
        return self.total_correlation / self.count

    def reset_state(self):
        self.total_correlation.assign(0)
        self.count.assign(0)
