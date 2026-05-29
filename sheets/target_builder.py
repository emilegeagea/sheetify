import numpy as np


def piano_roll_to_onset_frame(
    piano_roll: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_roll = piano_roll.astype(np.float32)

    padded = np.pad(frame_roll, ((0, 0), (1, 0)), mode="constant")

    onset_roll = (padded[:, 1:] > 0) & (padded[:, :-1] == 0)
    onset_roll = onset_roll.astype(np.float32)

    return onset_roll, frame_roll
