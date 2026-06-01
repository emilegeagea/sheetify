import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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

    plt.title("Piano Roll")
    plt.xlabel("Time (frames)")
    plt.ylabel("Piano key (0=A0, 87=C8)")

    plt.show()


def plot_history(history, metric_name: str = 'accuracy'):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))

    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title(metric_name)
    ax[1].plot(history.epoch, history.history[metric_name], label="Train " + metric_name)
    ax[1].plot(history.epoch, history.history["val_" + metric_name], label="Validation " + metric_name)
    ax[0].legend()
    ax[1].legend()

    plt.show()


def plot_pianoroll_overlay(
    pianoroll_true: np.ndarray,
    pianoroll_pred: np.ndarray,
) -> None:
    """
    Plot an overlay between ground-truth and predicted pianorolls.

    Encoding:
        0 = TN = white
        1 = FN = blue
        2 = FP = red
        3 = TP = black
    """

    assert pianoroll_true.shape == pianoroll_pred.shape, \
        "pianoroll_true and pianoroll_pred must have the same shape"

    overlay = pianoroll_true + 2 * pianoroll_pred

    cmap = ListedColormap([
        "white",  # TN
        "blue",   # FN
        "red",    # FP
        "black",  # TP
    ])

    norm = BoundaryNorm(
        boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5],
        ncolors=cmap.N,
    )

    plt.figure(figsize=(18, 8))

    im = plt.imshow(
        overlay,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    plt.title("Piano Roll Overlay")
    plt.xlabel("Time (frames)")
    plt.ylabel("Piano key (0=A0, 87=C8)")

    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["TN", "FN", "FP", "TP"])

    plt.tight_layout()
    plt.show()
