"""Precompute piano-roll clips (one MIDI parse per piece). Run from repo root:
python scripts/precompute_midi.py
"""

import math
from pathlib import Path

import numpy as np
import pretty_midi

from sheets.constants import *
from sheets.dataloader import MAESTRODataLoader
import sheets.helpers as helpers

import os
import asyncio

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from google.cloud import storage


BUCKET_NAME = "bobbybobster_sheetify"
DATA_ROOT = Path("./data")
OUT_ROOT = DATA_ROOT / "precomputed_rolls"


def roll_path(midi_path: str, clip_i: int) -> Path:
    rel = Path(midi_path).relative_to(DATA_ROOT / "midis").with_suffix("")
    return OUT_ROOT / rel.parent / f"{rel.name}_{clip_i:05d}.npy"


def slice_clip(roll: np.ndarray, start_sec: float) -> np.ndarray:
    t0 = int(start_sec * PIANO_ROLL_FS)
    t1 = t0 + int(CLIP_DURATION * PIANO_ROLL_FS)
    clip = roll[:, t0:t1]
    clip = clip[PIANO_MIN_PITCH : PIANO_MAX_PITCH + 1, :]
    clip = (clip > 0).astype(np.float32)
    return helpers.pad_or_trim_2d(clip, int(CLIP_DURATION * PIANO_ROLL_FS), axis=1)


async def upload_file(blob, path, semaphore):
    async with semaphore:
        try:
            await asyncio.to_thread(blob.upload_from_filename, str(path))
            print(f"✅ Uploaded file: {path}")
        except Exception as e:
            print(f"❌ Failed to upload {path}: {e}")


async def async_main(bucket):
    semaphore = asyncio.Semaphore(10)
    tasks = []
    # for year in [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]:
    for year in [2018]:
        print(f"📥 Precomputing year {year}")
        loader = MAESTRODataLoader(DATA_ROOT, year=year)
        for split in ("train", "validation", "test"):
            print(f"📥 Precomputing {split} splits")
            for pair in loader.get_pairs(split):
                n_clips = math.floor(pair["duration"] / CLIP_DURATION)
                if n_clips == 0:
                    continue
                roll = pretty_midi.PrettyMIDI(pair["midi_path"]).get_piano_roll(
                    fs=PIANO_ROLL_FS
                )
                for i in range(n_clips):
                    path = roll_path(pair["midi_path"], i)
                    if path.exists():
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, slice_clip(roll, i * CLIP_DURATION))
                    blob = bucket.blob(str(path))
                    tasks.append(upload_file(blob, path, semaphore))

            await asyncio.gather(*tasks)
            tasks.clear()
        print("✅ done")


if __name__ == "__main__":
    bucket_name = BUCKET_NAME
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print(f"{bucket_name=}")

    asyncio.run(async_main(bucket))
