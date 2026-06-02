import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from sheets.constants import *
from sheets.helpers import FlattenedFBetaScore


def conv_stack(x):
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D((1, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    return x


def squeeze_freq(x):
    x = tf.keras.layers.Permute((2, 1, 3))(x)
    shape = x.shape
    return tf.keras.layers.Reshape((shape[1], shape[2] * shape[3]))(x)


def map_to_target_time(x, n_time, name):
    x = tf.keras.layers.Permute((2, 1))(x)
    x = tf.keras.layers.Dense(n_time)(x)
    x = tf.keras.layers.Activation("sigmoid", name=name)(x)

    return x


def initialize_model(
    n_bins=N_BINS,
    n_frames=313,
    n_keys=N_PIANO_KEYS,
    n_time=int(CLIP_DURATION * PIANO_ROLL_FS),
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(n_bins, n_frames, 1))

    onset_x = conv_stack(inputs)
    onset_x = squeeze_freq(onset_x)
    onset_x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(onset_x)

    onset_seq = tf.keras.layers.Dense(
        n_keys,
        activation="sigmoid",
    )(onset_x)

    frame_x = conv_stack(inputs)
    frame_x = squeeze_freq(frame_x)

    frame_seq = tf.keras.layers.Dense(
        n_keys,
        activation="sigmoid",
    )(frame_x)

    combined = tf.keras.layers.Concatenate()([frame_seq, onset_seq])
    combined = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(combined)

    frame_seq = tf.keras.layers.Dense(n_keys)(combined)

    onset_output = map_to_target_time(onset_seq, n_time, "onset_output")
    frame_output = map_to_target_time(frame_seq, n_time, "frame_output")

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[onset_output, frame_output],
    )

    print("✅ Onsets & Frames model initialized")
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate=6e-4,
) -> tf.keras.Model:

    fbeta = FlattenedFBetaScore(beta=1.0, average='micro', name='fbeta')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "onset_output": "binary_crossentropy",
            "frame_output": "binary_crossentropy",
        },
        loss_weights={
            "onset_output": 1.0,
            "frame_output": 1.0,
        },
        metrics={
            "onset_output": [],
            "frame_output": [],
        },
    )

    print("✅ Model compiled")
    return model


def train_model(
    model: tf.keras.Model,
    X: np.ndarray,
    y_onset: np.ndarray,
    y_frame: np.ndarray,
    epochs: int = 10,
    batch_size: int = 16,
    patience: int = 2,
    validation_data=None,
    validation_split: float = 0.3,
) -> tuple[tf.keras.Model, dict]:
    print("\nTraining model...")

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    y = {
        "onset_output": y_onset,
        "frame_output": y_frame,
    }

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )

    print("✅ Model trained")
    return model, history
