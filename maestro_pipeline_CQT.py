"""
MAESTRO Audio-to-Sheet-Music Pipeline
======================================
Components:
  1. MAESTRODataLoader  - loads and pairs audio + MIDI from MAESTRO dataset
  2. CQTPreprocessor    - computes Constant-Q Transform features
  3. build_tf_dataset() - wires everything into a tf.data.Dataset

Requirements:
  pip install tensorflow librosa pretty_midi numpy pandas
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
import pretty_midi
import tensorflow as tf
from pathlib import Path
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

SAMPLE_RATE = 16_000          # Hz — MT3 standard
CLIP_DURATION = 10.0          # seconds per training chunk
HOP_LENGTH = 512              # CQT hop in samples → ~32ms at 16kHz
N_BINS = 84                   # 7 octaves × 12 semitones (full piano range)
BINS_PER_OCTAVE = 12
FMIN = librosa.note_to_hz("A0")  # 27.5 Hz — lowest piano key
PIANO_ROLL_FS = 100           # piano roll frames per second
PIANO_MIN_PITCH = 21          # MIDI A0
PIANO_MAX_PITCH = 108         # MIDI C8
N_PIANO_KEYS = PIANO_MAX_PITCH - PIANO_MIN_PITCH + 1  # 88


# ─────────────────────────────────────────────
# 1. DATA LOADER
# ─────────────────────────────────────────────

class MAESTRODataLoader:
    """
    Loads paired (audio, MIDI) clips from the MAESTRO v3 dataset.

    MAESTRO folder structure expected:
        maestro-v3.0.0/
            maestro-v3.0.0.json   ← metadata
            2004/
                *.wav
                *.midi
            2006/  ...

    Usage:
        loader = MAESTRODataLoader("path/to/maestro-v3.0.0")
        train_pairs = loader.get_pairs(split="train")
        audio, piano_roll = loader.load_pair(train_pairs[0])
    """

    def __init__(
        self,
        maestro_root: str,
        sr: int = SAMPLE_RATE,
        clip_duration: float = CLIP_DURATION,
        piano_roll_fs: int = PIANO_ROLL_FS,
    ):
        self.root = Path(maestro_root)
        self.sr = sr
        self.clip_duration = clip_duration
        self.clip_samples = int(sr * clip_duration)
        self.piano_roll_fs = piano_roll_fs

        # Load metadata JSON
        meta_path = self.root / "maestro-v3.0.0.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        self.metadata = pd.DataFrame(meta)

    def get_pairs(self, split: str = "train") -> List[dict]:
        """
        Returns list of {audio_path, midi_path, duration} dicts for a split.
        split: 'train' | 'validation' | 'test'
        """
        subset = self.metadata[self.metadata["split"] == split]
        pairs = []
        for _, row in subset.iterrows():
            pairs.append({
                "audio_path": str(self.root / row["audio_filename"]).replace(".wav", ".mp3"),
                "midi_path":  str(self.root / row["midi_filename"]),
                "duration":   row["duration"],
            })
        print(f"[MAESTRODataLoader] {split}: {len(pairs)} files found.")
        return pairs

    def load_pair(
        self,
        pair: dict,
        start_sec: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load one (audio_clip, piano_roll) pair.

        Args:
            pair: dict from get_pairs()
            start_sec: clip start time in seconds. If None, picks randomly.

        Returns:
            audio     : float32 array of shape (clip_samples,)
            piano_roll: float32 array of shape (N_PIANO_KEYS, piano_roll_frames)
                        values in [0, 1] — 1 means key is active
        """
        duration = pair["duration"]
        max_start = max(0.0, duration - self.clip_duration)

        if start_sec is None:
            start_sec = np.random.uniform(0, max_start)

        # ── Audio ──────────────────────────────────────
        audio, _ = librosa.load(
            pair["audio_path"],
            sr=self.sr,
            mono=True,
            offset=start_sec,
            duration=self.clip_duration,
        )
        audio = _pad_or_trim(audio, self.clip_samples)

        # ── Piano roll from MIDI ───────────────────────
        midi = pretty_midi.PrettyMIDI(pair["midi_path"])
        roll = midi.get_piano_roll(fs=self.piano_roll_fs)  # shape: (128, T)

        # Trim to clip window
        frame_start = int(start_sec * self.piano_roll_fs)
        frame_end = frame_start + int(self.clip_duration * self.piano_roll_fs)
        roll = roll[:, frame_start:frame_end]

        # Keep only piano range A0–C8, binarize (note on/off)
        roll = roll[PIANO_MIN_PITCH:PIANO_MAX_PITCH + 1, :]  # (88, T)
        roll = (roll > 0).astype(np.float32)

        # Pad time axis if needed
        target_frames = int(self.clip_duration * self.piano_roll_fs)
        roll = _pad_or_trim_2d(roll, target_frames, axis=1)

        return audio.astype(np.float32), roll


# ─────────────────────────────────────────────
# 2. CQT PREPROCESSOR
# ─────────────────────────────────────────────

class CQTPreprocessor:
    """
    Converts raw audio waveform → log-CQT spectrogram.

    Why CQT over mel-spectrogram?
      - Frequency bins align with semitones (logarithmic pitch scale)
      - Piano A0–C8 maps cleanly to 84 bins across 7 octaves
      - Better pitch resolution in low frequencies

    Output shape: (N_BINS, time_frames, 1)  ← channel-last for TF Conv2D
    """

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_bins: int = N_BINS,
        bins_per_octave: int = BINS_PER_OCTAVE,
        fmin: float = FMIN,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """
        Args:
            audio: float32 waveform, shape (clip_samples,)

        Returns:
            cqt_db: float32 array, shape (N_BINS, time_frames, 1)
                    normalized to [0, 1]
        """
        cqt = librosa.cqt(
            audio,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )
        # Convert to dB scale
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

        # Normalize to [0, 1]
        cqt_min, cqt_max = cqt_db.min(), cqt_db.max()
        if cqt_max > cqt_min:
            cqt_db = (cqt_db - cqt_min) / (cqt_max - cqt_min)
        else:
            cqt_db = np.zeros_like(cqt_db)

        # Add channel dim for Conv2D compatibility → (bins, frames, 1)
        return cqt_db[:, :, np.newaxis].astype(np.float32)

    def tf_compute(self, audio: tf.Tensor) -> tf.Tensor:
        """TensorFlow-compatible wrapper using tf.numpy_function."""
        cqt = tf.numpy_function(
            func=self.compute,
            inp=[audio],
            Tout=tf.float32,
        )
        # Set static shape so downstream layers know what to expect
        n_frames = int(CLIP_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1
        cqt.set_shape([N_BINS, n_frames, 1])
        return cqt


# ─────────────────────────────────────────────
# 3. TF DATASET BUILDER
# ─────────────────────────────────────────────

def build_tf_dataset(
    maestro_root: str,
    split: str = "train",
    batch_size: int = 16,
    shuffle_buffer: int = 200,
    prefetch: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """
    Full pipeline: MAESTRO files → batched tf.data.Dataset of (CQT, piano_roll).

    Args:
        maestro_root  : path to maestro-v3.0.0/ directory
        split         : 'train' | 'validation' | 'test'
        batch_size    : number of clips per batch
        shuffle_buffer: number of samples to shuffle

    Returns:
        tf.data.Dataset yielding:
            cqt        : float32 tensor (batch, N_BINS, time_frames, 1)
            piano_roll : float32 tensor (batch, 88, piano_roll_frames)
    """
    loader = MAESTRODataLoader(maestro_root)
    preprocessor = CQTPreprocessor()

    pairs = loader.get_pairs(split)

    def generator():
        for pair in pairs:
            try:
                audio, roll = loader.load_pair(pair)
                cqt = preprocessor.compute(audio)
                yield cqt, roll
            except Exception as e:
                print(f"[Warning] Skipping {pair['audio_path']}: {e}")
                continue

    # Infer output shapes
    n_frames = int(CLIP_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1
    roll_frames = int(CLIP_DURATION * PIANO_ROLL_FS)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(N_BINS, n_frames, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(N_PIANO_KEYS, roll_frames), dtype=tf.float32),
        ),
    )

    if split == "train":
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = (
        dataset
        .batch(batch_size, drop_remainder=True)
        .prefetch(prefetch)
    )

    return dataset


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def _pad_or_trim_2d(arr: np.ndarray, length: int, axis: int = 1) -> np.ndarray:
    current = arr.shape[axis]
    if current >= length:
        return np.take(arr, range(length), axis=axis)
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, length - current)
    return np.pad(arr, pad_width)


# ─────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    MAESTRO_ROOT = sys.argv[1] if len(sys.argv) > 1 else "./maestro-v3.0.0"

    print("=" * 50)
    print("Building train dataset...")
    train_ds = build_tf_dataset(MAESTRO_ROOT, split="train", batch_size=4)
    val_ds   = build_tf_dataset(MAESTRO_ROOT, split="validation", batch_size=4)

    print("\nFetching one batch...")
    for cqt_batch, roll_batch in train_ds.take(1):
        print(f"  CQT shape       : {cqt_batch.shape}")   # (4, 84, 313, 1)
        print(f"  Piano roll shape: {roll_batch.shape}")  # (4, 88, 1000)
        print(f"  CQT  min/max    : {cqt_batch.numpy().min():.3f} / {cqt_batch.numpy().max():.3f}")
        print(f"  Roll active %   : {roll_batch.numpy().mean() * 100:.1f}%")

    print("\n✅ Pipeline looks good — ready for fine-tuning.")
