import librosa
import numpy as np

from constants import *
from helpers import *


class Augmentor:
    """
    Musical-aware audio augmentations.

    Rules:
      - Time stretch / pitch shift must stay within musical range
        so piano roll labels remain valid
      - Noise and gain don't affect pitch → always safe
      - Pitch shift adjusts the target piano roll accordingly

    Usage:
        aug = Augmentor(pitch_shift_range=2, time_stretch_range=0.1)
        audio_aug, roll_aug = aug(audio, piano_roll)
    """

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        pitch_shift_range: int = 2,       # ± semitones
        time_stretch_range: float = 0.1,  # ± fraction (0.1 = ±10%)
        noise_std: float = 0.005,
        gain_range: float = 0.2,
        p_pitch_shift: float = 0.5,
        p_time_stretch: float = 0.5,
        p_noise: float = 0.3,
        p_gain: float = 0.5,
    ):
        self.sr = sr
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_range = time_stretch_range
        self.noise_std = noise_std
        self.gain_range = gain_range
        self.p_pitch_shift = p_pitch_shift
        self.p_time_stretch = p_time_stretch
        self.p_noise = p_noise
        self.p_gain = p_gain

    def __call__(
        self,
        audio: np.ndarray,
        piano_roll: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentations.

        Args:
            audio     : float32, shape (clip_samples,)
            piano_roll: float32, shape (88, piano_roll_frames)

        Returns:
            Augmented (audio, piano_roll) with same shapes.
        """
        audio = audio.copy()
        piano_roll = piano_roll.copy()

        # 1. Pitch shift ──────────────────────────────
        if np.random.rand() < self.p_pitch_shift:
            n_steps = np.random.randint(-self.pitch_shift_range, self.pitch_shift_range + 1)
            if n_steps != 0:
                audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
                piano_roll = self._shift_piano_roll(piano_roll, n_steps)

        # 2. Time stretch ─────────────────────────────
        if np.random.rand() < self.p_time_stretch:
            rate = np.random.uniform(
                1 - self.time_stretch_range,
                1 + self.time_stretch_range,
            )
            audio = librosa.effects.time_stretch(audio, rate=rate)
            # Re-trim/pad to original length
            audio = _pad_or_trim(audio, int(CLIP_DURATION * self.sr))
            # Stretch piano roll inversely
            piano_roll = self._stretch_piano_roll(piano_roll, rate)

        # 3. Gaussian noise ───────────────────────────
        if np.random.rand() < self.p_noise:
            noise = np.random.normal(0, self.noise_std, audio.shape).astype(np.float32)
            audio = audio + noise

        # 4. Random gain ──────────────────────────────
        if np.random.rand() < self.p_gain:
            gain = np.random.uniform(1 - self.gain_range, 1 + self.gain_range)
            audio = audio * gain

        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return audio, piano_roll

    def _shift_piano_roll(self, roll: np.ndarray, n_steps: int) -> np.ndarray:
        """Shift piano roll pitch rows by n_steps semitones."""
        shifted = np.zeros_like(roll)
        if n_steps > 0:
            shifted[n_steps:, :] = roll[:-n_steps, :]
        elif n_steps < 0:
            shifted[:n_steps, :] = roll[-n_steps:, :]
        else:
            shifted = roll
        return shifted

    def _stretch_piano_roll(self, roll: np.ndarray, rate: float) -> np.ndarray:
        """Re-sample piano roll time axis to match time-stretched audio."""
        n_keys, original_frames = roll.shape
        target_frames = int(original_frames / rate)
        # Use nearest-neighbor interpolation to avoid fractional note values
        indices = np.round(np.linspace(0, original_frames - 1, target_frames)).astype(int)
        indices = np.clip(indices, 0, original_frames - 1)
        stretched = roll[:, indices]
        return _pad_or_trim_2d(stretched, original_frames, axis=1)
