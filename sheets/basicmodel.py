# import tensorflow as tf
import keras
import numpy as np
from keras import optimizers
from keras.callbacks import EarlyStopping
import keras.saving
from colorama import Fore, Style

from sheets.constants import *
from sheets.helpers import FlattenedFBetaScore

keras.mixed_precision.set_global_policy('mixed_float16')

@keras.saving.register_keras_serializable(package="CustomLayers")
class PositionalEmbeddingAdder(keras.layers.Layer):
    def __init__(self, input_dim=400, output_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Instantiate inside __init__ so Keras tracks the weights natively
        self.pos_embedding_layer = keras.layers.Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )

    def call(self, inputs):
        # Dynamically calculate positions using keras.ops
        seq_len = keras.ops.shape(inputs)[1]
        positions = keras.ops.arange(0, seq_len, 1)

        # Add the positional embeddings to your inputs
        return inputs + self.pos_embedding_layer(positions)

    def get_config(self):
        # This tells Keras how to recreate the layer when loading
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        })
        return config


def initialize_model(
    n_bins=N_BINS,
    n_frames=313, # TODO: Should this also be a constant.
    n_keys=N_PIANO_KEYS,
    n_time=int(CLIP_DURATION * PIANO_ROLL_FS)
    ) -> keras.Model:
    inputs = keras.Input(shape=(n_bins, n_frames, 1))

    # --- CNN Block ---
    x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2,2))(x)  # Downsampling spatial dimensions
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2,2))(x)  # Tensor output shape here is (None, 21, 78, 256)
    x = keras.layers.Dropout(0.3)(x)

    # --- FIX: Natively collapse PITCH using Keras Layers ---
    # Global Average Pooling flattens spatial arrays, but setting keepdims=True
    # preserves the layout as (None, 1, 1, 256) which isn't what we want.
    # Instead, we use a standard Reshape to swap axis context, or average out Pitch.

    # We squash ONLY the Pitch axis (21 bins) down to 1 while leaving Time (78) alone.
    # To do this natively, we use Keras' functional operations API (keras.ops if Keras 3)
    # or a robust Lambda wrapper that plays perfectly with the standard compiler graph.
    x = keras.layers.Lambda(
        lambda t: keras.ops.mean(t, axis=1),
        output_shape=(78, 128)
        )(x) # Shape becomes: (None, 78, 256)


    x = PositionalEmbeddingAdder(input_dim=400, output_dim=128)(x)


    # --- Add Positional Tracking for the Transformer ---
    # 1. Instantiate the embedding layer OUTSIDE so Keras tracks its weights
    # TODO: Check whether this input_dim can be lowered or added dynamically

    # pos_embedding_layer = keras.layers.Embedding(input_dim=400, output_dim=256)

    # 2. Use a Lambda layer to wrap the raw TensorFlow ops (shape, tf.range)
    # x = keras.layers.Lambda(
        # lambda t: t + pos_embedding_layer(keras.ops.arange(0, keras.ops.shape(t)[1], 1)),
        # output_shape=(78, 256)
    # )(x)




    # --- Transformer Block ---
    x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = keras.layers.LayerNormalization()(x)

    # --- Learn the Time Mapping (No Blur) ---
    # Dynamically maps 78 audio steps up to 1000 piano-roll time targets.
    x = keras.layers.Permute((2, 1))(x)       # Shape: (None, 256, 78)
    x = keras.layers.Dense(n_time // 4)(x)         # Shape: (None, 256, 1000)
    x = keras.layers.Permute((2, 1))(x)       # Shape: (None, 1000, 256)
    # Scale up from 250 to 1000 frames using interpolation
    x = keras.layers.UpSampling1D(size=4)(x)  # Shape: (None, 1000, 256)

    # --- Output Block ---
    # Map 256 features directly to your 88 individual keys across every step.
    x = keras.layers.Dense(n_keys)(x)         # Shape: (None, 1000, 88)

    # Rearrange dimensions to deliver your target formatting: (None, 88, 1000)
    outputs = keras.layers.Permute((2, 1))(x)
    outputs = keras.layers.Activation('sigmoid')(outputs)

    model = keras.Model(inputs, outputs)

    print("✅ Model initialized")

    return model


@keras.saving.register_keras_serializable(package="MyMetrics")
def flattened_fbeta_score(y_true, y_pred):
    # Example logic: ensure calculation matches your objective
    # Replace this with your actual metric math
    y_true_flat = keras.ops.reshape(y_true, [-1, 88 * 1000])
    y_pred_flat = keras.ops.reshape(y_pred, [-1, 88 * 1000])
    return keras.metrics.binary_accuracy(y_true_flat, y_pred_flat)
flattened_fbeta_score.name = 'fbeta'


def compile_model(
    model: keras.Model,
    learning_rate=0.0005
    ) -> keras.Model:
    # If your data has 10x more zeros than ones, set pos_weight to roughly 10.0
    # loss_fn = keras.losses.BinaryCrossentropy(
        # from_logits=False,
        # reduction="sum_over_batch_size"
    # )

    # Note: In standard Keras, to inject a raw 'pos_weight', you can also wrap
    # the native TensorFlow function easily:
    # def weighted_bce(y_true, y_pred):
        # return tf.nn.weighted_cross_entropy_with_logits(
            # labels=y_true,
            # logits=tf.math.log(y_pred / (1.0 - y_pred + 1e-7) + 1e-7), # convert back to logits safely
            # pos_weight=10.0
        # )

    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[flattened_fbeta_score],
    )

    print("✅ Model compiled")

    return model

def train_model(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 10,
    batch_size: int = 256,
    patience: int = 2,
    validation_data=None, # overrides validation_split
    validation_split: float = 0.3
    ) -> tuple[keras.Model, dict]:
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
