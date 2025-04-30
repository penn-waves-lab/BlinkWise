"""
Labeling of blink phases may appear in the two following identical forms:
spiky labels: which only marks the beginning of each phase, and all other data points are marked as -1 (NOT_INITIALIZED)
continuous labels: which marks each data point with the corresponding phase label (denser).

example:
spiky_labels = [-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 0, -1, -1]
continuous_labels = [-1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0]

the following functions are used to convert between the two forms
"""

import numpy as np

from src.core import blink_defs


def from_spiky_labels_to_continuous_labels(spiky_labels: np.ndarray[int]) -> np.ndarray[int]:
    """Converts spiky labels to continuous labels. See module docstring for more information."""
    spiky_labels_to_convert = spiky_labels.copy()
    if spiky_labels_to_convert[0] == blink_defs.NOT_INITIALIZED:
        spiky_labels_to_convert[0] = blink_defs.NON_BLINKING

    continuous_labels = np.zeros((spiky_labels.shape[0],), dtype=int)

    # we have included additional data. these are added to make sure all data are covered with labels
    if spiky_labels[0] == -1:
        continuous_labels[0] = blink_defs.NON_BLINKING
    if spiky_labels[-1] == -1:
        continuous_labels[-1] = blink_defs.NON_BLINKING

    # mark labels
    all_phase_start_indices = np.where(spiky_labels != -1)[0]
    for i in range(all_phase_start_indices.shape[0] - 1):
        phase_start_time = all_phase_start_indices[i]
        phase_end_time = all_phase_start_indices[i + 1]
        continuous_labels[phase_start_time:phase_end_time] = spiky_labels[
            all_phase_start_indices[i]
        ]

    return continuous_labels


def from_continuous_labels_to_spiky_labels(continuous_labels: np.ndarray[int]) -> np.ndarray[int]:
    """Converts continuous labels to spiky labels. See module docstring for more information."""
    converted_spiky_labels = np.full_like(continuous_labels, blink_defs.NOT_INITIALIZED)
    label_change_locations = np.where(np.diff(continuous_labels) != 0)[0] + 1
    for i in label_change_locations:
        converted_spiky_labels[i] = continuous_labels[i]
    return converted_spiky_labels
