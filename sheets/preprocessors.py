import librosa
import os
import math
import numpy as np
import glob
import matplotlib.pyplot as plt
import pretty_midi
import tensorflow as tf
from pathlib import Path
import constants.py


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


class MELPreprocessor:
    def __init__(self,
                 slice_duration: float = CLIP_DURATION,
                 n_mels: int = N_MELS,
                 n_fft: int = 2048,
                 hop_length: int = HOP_LENGTH,
                 sr: int = SAMPLE_RATE):
        self.slice_duration = slice_duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.clip_samples = int(sr * slice_duration)

    def compute_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Converts raw audio waveform → normalized log-mel spectrogram.

        Args:
            audio: float32 waveform, shape (clip_samples,)

        Returns:
            mel_db: float32 array, shape (N_MELS, time_frames, 1)
                    normalized to [0, 1] — matches CQT pipeline output format
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1] — same as CQT pipeline
        mel_min, mel_max = mel_db.min(), mel_db.max()
        if mel_max > mel_min:
            mel_db = (mel_db - mel_min) / (mel_max - mel_min)
        else:
            mel_db = np.zeros_like(mel_db)

        # Add channel dim → (n_mels, frames, 1)
        return mel_db[:, :, np.newaxis].astype(np.float32)

    def load_pair(
        self,
        audio_path: str,
        midi_path: str,
        start_sec: float = None,
    ):
        """
        Load one (mel_spectrogram, piano_roll) pair.
        Same interface as MAESTRODataLoader.load_pair() for easy comparison.

        Returns:
            mel  : float32 (N_MELS, time_frames, 1)
            roll : float32 (88, piano_roll_frames)
        """
        # Get duration for random start
        duration = librosa.get_duration(path=audio_path)
        max_start = max(0.0, duration - self.slice_duration)
        if start_sec is None:
            start_sec = np.random.uniform(0, max_start)

        # Load audio — resample to 16kHz mono (same as CQT pipeline)
        audio, _ = librosa.load(
            audio_path,
            sr=self.sr,
            mono=True,
            offset=start_sec,
            duration=self.slice_duration,
        )
        audio = self._pad_or_trim(audio)

        # Compute mel spectrogram
        mel = self.compute_mel(audio)

        # Load MIDI → piano roll
        midi = pretty_midi.PrettyMIDI(midi_path)
        roll = midi.get_piano_roll(fs=PIANO_ROLL_FS)
        frame_start = int(start_sec * PIANO_ROLL_FS)
        frame_end = frame_start + int(self.slice_duration * PIANO_ROLL_FS)
        roll = roll[:, frame_start:frame_end]
        roll = roll[PIANO_MIN_PITCH:PIANO_MAX_PITCH + 1, :]
        roll = (roll > 0).astype(np.float32)
        roll = self._pad_or_trim_roll(roll)

        return mel, roll

    def build_tf_dataset(
        self,
        maestro_root: str,
        split: str = "train",
        batch_size: int = 16,
        shuffle_buffer: int = 200,
        prefetch: int = tf.data.AUTOTUNE,
    ) -> tf.data.Dataset:
        """
        Same interface as build_tf_dataset() in maestro_pipeline_CQT.py.
        Produces (mel_spectrogram, piano_roll) batches instead of (CQT, piano_roll).
        """
        import json
        import pandas as pd

        meta_path = next(Path(maestro_root).glob("*.json"))
        with open(meta_path) as f:
            meta = json.load(f)
        df = pd.DataFrame(meta)
        subset = df[df["split"] == split]

        pairs = []
        for _, row in subset.iterrows():
            pairs.append({
                # "audio_path": str(Path(maestro_root) / row["audio_filename"]).replace(".wav", ".mp3")
                "audio_path": "midi.mp3",
                # "midi_path":  str(Path(maestro_root) / row["midi_filename"]),
                "midi_path" : "midi.mid",
                # "duration":   row["duration"],
                "duration" : 26
            })
        pairs = pairs[:2]

        print(f"[MelPipeline] {split}: {len(pairs)} files found.")

        pipeline = self

        def generator():
            for pair in pairs:
                try:
                    mel, roll = pipeline.load_pair(pair["audio_path"], pair["midi_path"])
                    yield mel, roll
                except Exception as e:
                    print(f"[Warning] Skipping {pair['audio_path']}: {e}")
                    continue

        n_frames = int(self.slice_duration * self.sr / self.hop_length) + 1
        roll_frames = int(self.slice_duration * PIANO_ROLL_FS)

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.n_mels, n_frames, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(N_PIANO_KEYS, roll_frames),  dtype=tf.float32),
            ),
        )

        if split == "train":
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(prefetch)
        return dataset

    def save_mel_spec_images(
        self,
        mel_spec_mat: np.ndarray,
        file_name: str,
        output_dir: str = os.getcwd() + '/mel_spec_images',
    ) -> None:
        """Save mel spectrogram slices as PNG images."""
        os.makedirs(output_dir, exist_ok=True)
        file_basename = os.path.splitext(os.path.basename(file_name))[0]
        for idx in range(mel_spec_mat.shape[-1]):
            base_filename = f"{file_basename}_Split_Part_{idx+1}"
            plt.imsave(output_dir + '/' + base_filename + '.png', mel_spec_mat[:, :, idx])

    def run_from_folder(self, input_folder: str):
        """
        Yields (file_path, mel_spec_array) for each .mp3 in folder.
        Same interface as before.
        """
        mp3_files = glob.glob(os.path.join(input_folder, "*.mp3"))
        for index, current_file in enumerate(mp3_files):
            print(f"[{index+1}/{len(mp3_files)}] Processing: {os.path.basename(current_file)}")
            yield current_file, self.split_audio(current_file)

    def split_audio(self, file_path: str) -> np.ndarray:
        """Original method preserved — splits full audio into mel spectrogram slices."""
        total_duration = librosa.get_duration(path=file_path)
        total_slices = math.floor(total_duration / self.slice_duration)
        n_frames = (self.slice_duration * self.sr) // self.hop_length + 1
        all_mel_specs = np.zeros(shape=(self.n_mels, int(n_frames), total_slices))

        for i in range(total_slices):
            start_time = i * self.slice_duration
            remaining_time = total_duration - start_time
            current_duration = min(remaining_time, self.slice_duration)
            y, _ = librosa.load(file_path, sr=self.sr, offset=start_time, duration=current_duration)
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_min, mel_max = mel_db.min(), mel_db.max()
            if mel_max > mel_min:
                mel_db = (mel_db - mel_min) / (mel_max - mel_min)
            all_mel_specs[:, :, i] = mel_db

        return all_mel_specs

    # ── Helpers ────────────────────────────────────────────────────
    def _pad_or_trim(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) >= self.clip_samples:
            return audio[:self.clip_samples]
        return np.pad(audio, (0, self.clip_samples - len(audio)))

    def _pad_or_trim_roll(self, roll: np.ndarray) -> np.ndarray:
        target = int(self.slice_duration * PIANO_ROLL_FS)
        if roll.shape[1] >= target:
            return roll[:, :target]
        return np.pad(roll, ((0, 0), (0, target - roll.shape[1])))
