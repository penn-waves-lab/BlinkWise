import io
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Sequence, Optional, Union

import matplotlib
# Set the backend to 'Agg' for non-interactive plotting
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.core import blink_defs, validate_literal_args

if os.environ.get("BLINKWISE_USE_TFMOT_KERAS", "0") == "0":
    from tensorflow import keras
else:
    from tensorflow_model_optimization.python.core.keras.compat import keras

# ===================
# helper functions
# ===================

def label_to_color(label: int) -> str:
    if label == blink_defs.NON_BLINKING:
        return "#93BEDF"
    elif label == blink_defs.CLOSING:
        return "#8EF9F3"
    elif label == blink_defs.INTERPHASE:
        return "#8377D1"
    elif label == blink_defs.REOPENING:
        return "#5EFC8D"
    elif label == -1:
        return "#FFBFB0"
    else:
        return "#DBD4D3"


def convert_to_image(fig):
    """
    Convert a matplotlib figure to a tensor suitable for TensorBoard logging.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)  # Add batch dimension
    return image


# ===================
# main visualizer classes
# ===================


class BaseVisualizer(keras.callbacks.Callback, ABC):
    def __init__(self, log_dir: Union[str, Path], data: tuple, tag, freq, ncols: int):
        """
        Base class for visualizers. They generate matplotlib figures and log them to TensorBoard.

        Args:
            log_dir: The directory where the logs will be saved.
            data: The data to visualize.
            tag: The tag to use when logging the image.
            freq: The frequency at which to log the image. Measured in epochs.
            ncols: The number of columns in the grid of images.
        """
        super().__init__()
        self.log_dir = str(log_dir)
        self.data = data
        self.tag = tag
        self.freq = freq
        self.ncols = ncols

        self.file_writer = tf.summary.create_file_writer(self.log_dir)

    @abstractmethod
    def visualize(self, x, y, predictions):
        """
        Generate a matplotlib figure from data and predictions.

        Must be implemented by subclasses.
        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            x, y = self.data
            predictions = self.model.predict(x)

            # Generate matplotlib figure using abstract method
            fig = self.visualize(x, y, predictions)
            plt.tight_layout()

            # Convert figure to tensor image
            image = convert_to_image(fig)

            # Log the image using TensorBoard
            with self.file_writer.as_default():
                tf.summary.image(self.tag, image, step=epoch)


class CurvePredictionVisualizer(BaseVisualizer):
    def __init__(self,
                 log_dir: Union[str, Path], data: tuple, tag: str, freq: int, ncols: int = 4,
                 range_bin_to_plot: Optional[int] = None, ant_to_plot=0, show_input=True
                 ):
        """
        Visualizer for curve prediction tasks.

        Args:
            log_dir: The directory where the logs will be saved.
            data: The data to visualize. Tuple of (x, y).
            tag: The tag to use when logging the image.
            freq: The frequency at which to log the image. Measured in epochs.
            ncols: The number of columns in the grid of images. Default is 4.
            range_bin_to_plot: The range bin of input data to plot. If None, the middle range bin is used.
            ant_to_plot: The antenna of input data to plot. Default is 0.
            show_input: Whether to show the input signal in the plot. Default is True.
        """
        super().__init__(log_dir, data, tag, freq, ncols)

        self.x_shape = data[0].shape
        self.y_shape = data[1].shape

        if range_bin_to_plot is None:
            range_bin_to_plot = self.x_shape[2] // 2
        self.range_bin_to_plot = range_bin_to_plot
        self.ant_to_plot = ant_to_plot

        self.show_input = show_input

        self.to_plot = self.prepare_signal_to_plot(self.data[0])

    def prepare_signal_to_plot(self, x):
        def check_dims(expected_dim):
            assert len(self.x_shape) == expected_dim, (f"Expected n dimensions is {expected_dim} (including batch), "
                                                       f"got {len(self.x_shape)}.")
        # B x T x R x A
        check_dims(4)
        return x[:, :, self.range_bin_to_plot, self.ant_to_plot]

    @staticmethod
    def _visualize_blink_phases_as_spans(ax, label_to_plot, ymin=0., ymax=1., alpha=1.):
        locs = np.concatenate((
            [0],
            np.where(np.diff(label_to_plot, prepend=label_to_plot[0]) != 0)[0],
            [label_to_plot.shape[0]]
        ))
        for j in range(locs.shape[0] - 1):
            start_idx = locs[j]
            end_idx = locs[j + 1]
            ax.axvspan(start_idx, end_idx, ymin, ymax, color=label_to_color(label_to_plot[start_idx]), alpha=alpha)

    @staticmethod
    def _visualize_blink_phases_as_lines(ax, label_to_plot, ymin=0., ymax=1., lw=5., alpha=1.):
        for i, label in enumerate(np.unique(label_to_plot)):
            indices = np.where(label_to_plot == label)[0]
            ax.scatter(
                indices, np.full_like(indices, (ymin + ymax) / 2, dtype=float),
                s=lw, c=label_to_color(round(label)), alpha=alpha,
            )

        transitions = np.where(np.diff(label_to_plot) != 0)[0] + 1
        for t in transitions:
            ax.axvline(t, ymin, ymax, c=label_to_color(round(label_to_plot[t])), lw=lw, alpha=alpha)

    def visualize(self, x, y, predictions):
        """
        T - Uniform length of a blink
        x dimensions: B x T x R x A.
        predictions: for regression: B x T.
        """
        n_batches = predictions.shape[0]

        ncols = min(self.ncols, n_batches)
        nrows = math.ceil(n_batches / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
        axes = axes.ravel()

        for b in range(min(n_batches, axes.shape[0])):
            ax = axes[b]
            # visualize the input
            if self.show_input:
                ax.plot(self.to_plot[b], label="input")
                ax.legend()

            pred_label_to_plot = predictions[b]
            true_label_to_plot = y[b]

            # visualize fitted curve.
            ax2 = ax.twinx()
            if np.any(y < 0):
                ax2.set_ylim((-1, 1))
                ax2.axhline(0, linestyle="--", alpha=0.5)
            else:
                ax2.set_ylim((0, 1))
            ax2.plot(pred_label_to_plot, c="#FF1B1C", label="prediction")
            ax2.plot(true_label_to_plot, c="#91CB3E", label="gt")
            ax2.legend()

        return fig


@validate_literal_args
def visualizer_factory(
        visualizer_type: Literal["unet"],
        log_dir: Union[str, Path],
        data_subset,
        tag: str,
        freq: int = 5,
        ncols: int = 4,
        **kwargs
) -> BaseVisualizer:
    """
    Factory function to create visualizers.

    Args:
        visualizer_type: The type of visualizer to create. Supported: unet.
        log_dir: The directory where the logs will be saved.
        data_subset: The data to visualize.
        tag: The tag to use when logging the image.
        freq: The frequency at which to log the image. Measured in epochs. Default is 5.
        ncols: The number of columns in the grid of images. Default is 4.

    Returns:
        The visualizer callback.
    """
    if visualizer_type == "unet":
        return CurvePredictionVisualizer(
            log_dir, data_subset, tag, freq, ncols=ncols,
            range_bin_to_plot=kwargs.get("range_bin_to_plot", None),
            ant_to_plot=kwargs.get("ant_to_plot", 0),
            show_input=kwargs.get("show_input", True),
        )
    else:
        raise ValueError(f"Unknown visualizer type. Got {visualizer_type}, supported: unet.")

# ===================
# loss weight scheduler
# ===================

class WeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_weights: Sequence[tf.Variable], loc_to_change=-1, step=0.01, upper_bound=0.5):
        super().__init__()
        self.weights = initial_weights
        self.n_weights = len(self.weights)

        self.loc_to_change = loc_to_change % self.n_weights
        self.step = step
        if self.n_weights == 1:
            self.decrease_step = self.step
        else:
            self.decrease_step = self.step / (self.n_weights - 1)
        self.upper_bound = upper_bound

    def on_epoch_begin(self, epoch, logs=None):
        alpha = keras.backend.get_value(self.weights[self.loc_to_change])
        if alpha <= self.upper_bound:
            keras.backend.set_value(self.weights[self.loc_to_change], alpha + self.step)

            for i in range(self.n_weights):
                if i != self.loc_to_change:
                    keras.backend.set_value(self.weights[i], keras.backend.get_value(self.weights[i]) - self.decrease_step)
