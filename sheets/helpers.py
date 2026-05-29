import numpy as np
import tensorflow as tf

from sheets.constants import *

class FlattenedFBetaScore(tf.keras.metrics.FBetaScore):
    """
    Usage:
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[FlattenedFBetaScore(beta=1.0)])
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Dynamically reshape 3D tensors (None, 88, 1000) to 2D (None, 88000)
        y_true_flat = tf.reshape(y_true, [-1, 88 * 1000])
        y_pred_flat = tf.reshape(y_pred, [-1, 88 * 1000])

        return super().update_state(y_true_flat, y_pred_flat, sample_weight)


def pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def pad_or_trim_2d(arr: np.ndarray, length: int, axis: int = 1) -> np.ndarray:
    current = arr.shape[axis]
    if current >= length:
        return np.take(arr, range(length), axis=axis)
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, length - current)
    return np.pad(arr, pad_width)


def pad_or_trim_roll(roll: np.ndarray) -> np.ndarray:
    target = int(self.slice_duration * PIANO_ROLL_FS)
    if roll.shape[1] >= target:
        return roll[:, :target]
    return np.pad(roll, ((0, 0), (0, target - roll.shape[1])))
