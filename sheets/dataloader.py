import librosa
import pretty_midi
import numpy as np
import pandas as pd
import json

from pathlib import Path
from typing import Optional, Literal
from constants import *
from helpers import *


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
        maestro_root: str = './data',
        year: int | list[int] | Literal['all'] = 2018,
        sr: int = SAMPLE_RATE,
        clip_duration: float = CLIP_DURATION,
        piano_roll_fs: int = PIANO_ROLL_FS,
    ):
        self.root = Path(maestro_root)
        self.year = year
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

        # Ensure we always store a list-like in self.year
        if isinstance(year, int):
            self.year = [year]
        if year == 'all':
            self.year = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]


    def get_pairs(
        self,
        split: str = "train",
        limit: int | None = None,
        ) -> list[dict]:
        """
        Returns list of {audio_path, midi_path, duration} dicts for a split.
        split: 'train' | 'validation' | 'test'
        """
        subset = self.metadata[self.metadata["split"] == split]
        # Only get data for a specific (list of) year(s) or all
        subset = subset[self.metadata['year'].isin(self.year)]
        pairs = []
        for _, row in subset.iterrows():
            pairs.append({
                "audio_path": str(self.root / 'midis' / row['audio_filename'].replace('.wav', '.mp3')),
                "midi_path":  str(self.root / 'mp3s'  / row["midi_filename"]),
                "duration":   row["duration"],
            })

        if limit is not None:
            pairs = pairs[:limit]

        print(f"[MAESTRODataLoader] {split}: {len(pairs)} files found.")
        return pairs

    def load_pair(
        self,
        pair: dict,
        start_sec: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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
