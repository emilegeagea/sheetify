import tensorflow as tf
import numpy as np
from keras import optimizers
from keras.callbacks import EarlyStopping
from colorama import Fore, Style

from sheets.constants import *

def initialize_model(
    n_bins=N_BINS,
    n_frames=313, # TODO: Should this also be a constant.
    n_keys=N_PIANO_KEYS,
    n_time=int(CLIP_DURATION * PIANO_ROLL_FS)
    ) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(n_bins, n_frames, 1))

    # --- CNN Block ---
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)  # Downsampling spatial dimensions
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)  # Tensor output shape here is (None, 21, 78, 256)
    x = tf.keras.layers.Dropout(0.3)(x)

    # --- FIX: Natively collapse PITCH using Keras Layers ---
    # Global Average Pooling flattens spatial arrays, but setting keepdims=True
    # preserves the layout as (None, 1, 1, 256) which isn't what we want.
    # Instead, we use a standard Reshape to swap axis context, or average out Pitch.

    # We squash ONLY the Pitch axis (21 bins) down to 1 while leaving Time (78) alone.
    # To do this natively, we use Keras' functional operations API (tf.keras.ops if Keras 3)
    # or a robust Lambda wrapper that plays perfectly with the standard compiler graph.
    x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x) # Shape becomes: (None, 78, 256)

    # --- Add Positional Tracking for the Transformer ---
    # Ensures the attention block knows the exact order of your 78 chronological slices.
    seq_len = 78
    pos_indices = tf.range(start=0, limit=seq_len, delta=1)
    pos_embed = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=256)(pos_indices)
    x = x + pos_embed  # Pure (None, 78, 256) sequential data

    # --- Transformer Block ---
    x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = tf.keras.layers.LayerNormalization()(x)

    # --- Learn the Time Mapping (No Blur) ---
    # Dynamically maps 78 audio steps up to 1000 piano-roll time targets.
    x = tf.keras.layers.Permute((2, 1))(x)       # Shape: (None, 256, 78)
    x = tf.keras.layers.Dense(n_time)(x)         # Shape: (None, 256, 1000)
    x = tf.keras.layers.Permute((2, 1))(x)       # Shape: (None, 1000, 256)

    # --- Output Block ---
    # Map 256 features directly to your 88 individual keys across every step.
    x = tf.keras.layers.Dense(n_keys)(x)         # Shape: (None, 1000, 88)

    # Rearrange dimensions to deliver your target formatting: (None, 88, 1000)
    outputs = tf.keras.layers.Permute((2, 1))(x)
    outputs = tf.keras.layers.Activation('sigmoid')(outputs)

    model = tf.keras.Model(inputs, outputs)

    print("✅ Model initialized")

    return model

def compile_model(
    model: tf.keras.Model,
    learning_rate=0.0005
    ) -> tf.keras.Model:
    # If your data has 10x more zeros than ones, set pos_weight to roughly 10.0
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        reduction="sum_over_batch_size"
    )

    # Note: In standard Keras, to inject a raw 'pos_weight', you can also wrap
    # the native TensorFlow function easily:
    def weighted_bce(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=tf.math.log(y_pred / (1.0 - y_pred + 1e-7) + 1e-7), # convert back to logits safely
            pos_weight=10.0
        )

    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=weighted_bce,
        optimizer=optimizer,
        metrics=['accuracy'],
    )

    print("✅ Model compiled")

    return model

def train_model(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    epochs=10,
    batch_size=256,
    patience=2,
    validation_data=None, # overrides validation_split
    validation_split=0.3
    ) -> tuple[tf.keras.Model, dict]:
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with min val Accuracy: {round(np.min(history.history['accuracy']), 2)}")

    return model, history
