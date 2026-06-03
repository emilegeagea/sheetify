import librosa
import pretty_midi
import numpy as np
import pandas as pd
import json

from pathlib import Path
from typing import Optional, Literal

from sheets.constants import *
import sheets.helpers as helpers
from sheets.preprocessors import Preprocessor


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
        Precomputed (optional): data/precomputed_rolls|precomputed_cqt/<midi-rel>_<clip>.npy
    """

    def __init__(
        self,
        maestro_root: str = './data',
        year: int | list[int] | Literal['all'] = 2018,
        limit: int | None = None,
        sr: int = SAMPLE_RATE,
        clip_duration: float = CLIP_DURATION,
        piano_roll_fs: int = PIANO_ROLL_FS,
    ):
        self.root = Path(maestro_root)
        self.year = year
        self.limit = limit
        self.sr = sr
        self.clip_duration = clip_duration
        self.clip_samples = int(sr * clip_duration)
        self.piano_roll_fs = piano_roll_fs
        self.precomputed_rolls_root = self.root / "precomputed_rolls"
        self.precomputed_cqt_root = self.root / "precomputed_cqt"

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
        elif isinstance(year, str):
            self.year = [int(year)]


    def get_pairs(
        self,
        split: str = "train",
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
                "audio_path": str(self.root / 'mp3s' / row['audio_filename'].replace('.wav', '.mp3')),
                "midi_path": str(self.root / 'midis' / row["midi_filename"]),
                "duration": row["duration"],
            })

        if self.limit is not None:
            pairs = pairs[:self.limit]

        print(f"✅ [MAESTRODataLoader] {split}: {len(pairs)} files found.")
        return pairs


    def _precomputed_roll_path(self, pair: dict, start_sec: float) -> Path:
        clip_i = int(start_sec / self.clip_duration)
        rel = Path(pair["midi_path"]).relative_to(self.root / "midis").with_suffix("")
        return self.precomputed_rolls_root / rel.parent / f"{rel.name}_{clip_i:05d}.npy"

    def _precomputed_cqt_path(self, pair: dict, start_sec: float) -> Path:
        clip_i = int(start_sec / self.clip_duration)
        rel = Path(pair["audio_path"]).relative_to(self.root / "mp3s").with_suffix("")
        return self.precomputed_cqt_root / rel.parent / f"{rel.name}_{clip_i:05d}.npy"



    def load_CQT(
        self,
        pair: dict,
        start_sec: float,
        preprocessor: Preprocessor,
    ) -> np.ndarray:
        """Load only a (precomputed) spectrogram"""
        cqt_path = self._precomputed_cqt_path(pair, start_sec)
        if cqt_path.exists():
            # print(f'🔋 Dataloader: Precomputed CQT found : {pair["audio_path"]=}')
            cqt = np.load(cqt_path).astype(np.float32)
        else:
            print(f'🪫 Dataloader: No precomputed CQT : {pair["audio_path"]=}')
            audio, _ = librosa.load(
                pair["audio_path"],
                sr=self.sr,
                mono=True,
                offset=start_sec,
                duration=self.clip_duration,
            )
            audio = helpers.pad_or_trim(audio, self.clip_samples)
            cqt = preprocessor.compute(audio)
        return cqt


    def load_roll(
        self,
        pair: dict,
        start_sec: float,
    ) -> np.ndarray:
        """Load only a (precomputed) pianoroll."""
        roll_path = self._precomputed_roll_path(pair, start_sec)
        if roll_path.exists():
            # print(f'🔋 Dataloader: Precomputed pianrolls found : {pair["midi_path"]=}')
            roll = np.load(roll_path).astype(np.float32)
        else:
            print(f'🪫 Dataloader: No precomputed pianorolls : {pair["midi_path"]=}')
            midi = pretty_midi.PrettyMIDI(pair["midi_path"])
            roll = midi.get_piano_roll(fs=self.piano_roll_fs)
            frame_start = int(start_sec * self.piano_roll_fs)
            frame_end = frame_start + int(self.clip_duration * self.piano_roll_fs)
            roll = roll[:, frame_start:frame_end]
            roll = roll[PIANO_MIN_PITCH : PIANO_MAX_PITCH + 1, :]
            roll = (roll > 0).astype(np.float32)
            target_frames = int(self.clip_duration * self.piano_roll_fs)
            roll = helpers.pad_or_trim_2d(roll, target_frames, axis=1)

        return roll



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

        # MP3 / AUDIO
        audio, _ = librosa.load(
            pair["audio_path"],
            sr=self.sr,
            mono=True,
            offset=start_sec,
            duration=self.clip_duration,
        )
        audio = helpers.pad_or_trim(audio, self.clip_samples)


        # MIDI / PIANOROLL
        roll_path = self._precomputed_roll_path(pair, start_sec)
        if roll_path.exists():
            # print(f'🔋 Dataloader: Precomputed pianrolls found : {pair["midi_path"]=}')
            roll = np.load(roll_path).astype(np.float32)
        else:
            print(f'🪫 Dataloader: No precomputed pianorolls : {pair["midi_path"]=}')
            midi = pretty_midi.PrettyMIDI(pair["midi_path"])
            roll = midi.get_piano_roll(fs=self.piano_roll_fs)
            frame_start = int(start_sec * self.piano_roll_fs)
            frame_end = frame_start + int(self.clip_duration * self.piano_roll_fs)
            roll = roll[:, frame_start:frame_end]
            roll = roll[PIANO_MIN_PITCH : PIANO_MAX_PITCH + 1, :]
            roll = (roll > 0).astype(np.float32)
            target_frames = int(self.clip_duration * self.piano_roll_fs)
            roll = helpers.pad_or_trim_2d(roll, target_frames, axis=1)

        return audio.astype(np.float32), roll
