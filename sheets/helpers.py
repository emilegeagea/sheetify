import numpy as np
import matplotlib.pyplot as plt

from sheets.constants import *

def plot_pianoroll_in_vs_out(
    ground_truth: np.ndarray,
    prediction: np.ndarray
    ) -> None:
    # TODO: Add switch for having time in frames vs seconds
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    axes[0].set_title("Piano Roll (ground truth)")
    axes[0].imshow(ground_truth, aspect="auto", origin="lower", cmap="Blues")
    axes[0].set_xlabel("Time (frames)")
    axes[0].set_ylabel("Piano key (0=A0, 87=C8)")

    axes[1].set_title("Model output")
    axes[1].imshow(prediction, aspect="auto", origin="lower", cmap="Blues")
    axes[1].set_xlabel("Time (frames)")
    axes[1].set_ylabel("Piano key (0=A0, 87=C8)")

    plt.tight_layout()
    plt.show()


def plot_pianoroll(piano_roll: np.ndarray) -> None:
    plt.figure(figsize=(15, 4))
    plt.imshow(piano_roll, aspect="auto", origin="lower", cmap="Blues")

    plt.title("Piano Roll (ground truth)")
    plt.xlabel("Time (frames)")
    plt.ylabel("Piano key (0=A0, 87=C8)")

    plt.show()


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
