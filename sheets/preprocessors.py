import librosa
import os
import math
import numpy as np
import glob
import matplotlib.pyplot as plt
import pretty_midi
import tensorflow as tf
from pathlib import Path

from sheets.constants import *
from sheets.helpers import *


class Preprocessor:
    def compute(self, audio: np.ndarray) -> np.ndarray:
        pass


class CQTPreprocessor(Preprocessor):
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
        # Convert to dB scale, with CQT we should use amplitude_to_db
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


class MELPreprocessor(Preprocessor):
    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_fft: int = 2048,
        n_mels: int = N_MELS,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels

    def compute(self, audio: np.ndarray) -> np.ndarray:
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
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
        )
        # Convert to dB scale, as we are using the default power=2.0 in calculating
        # the melspectrogram, we should use power_to_db
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_min, mel_max = mel_db.min(), mel_db.max()
        if mel_max > mel_min:
            mel_db = (mel_db - mel_min) / (mel_max - mel_min)
        else:
            mel_db = np.zeros_like(mel_db)

        # Add channel dim for Conv2D compatibility → (n_mels, frames, 1)
        return mel_db[:, :, np.newaxis].astype(np.float32)


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
        self.slice_duration = CLIP_DURATION
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
