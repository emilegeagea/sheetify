"""Precompute CQT spectrogram clips. Run from repo root:
python scripts/precompute_cqt.py
"""

import math
from pathlib import Path

import librosa
import numpy as np
import asyncio
import warnings

from sheets.constants import *
from sheets.dataloader import MAESTRODataLoader
from sheets.preprocessors import CQTPreprocessor
import sheets.helpers as helpers

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from google.cloud import storage


BUCKET_NAME = "bobbybobster_sheetify"
DATA_ROOT = Path("./data")
OUT_ROOT = DATA_ROOT / "precomputed_cqt"
preprocessor = CQTPreprocessor()


def cqt_path(mp3_path: str, clip_i: int) -> Path:
    rel = Path(mp3_path).relative_to(DATA_ROOT / "mp3s").with_suffix("")
    return OUT_ROOT / rel.parent / f"{rel.name}_{clip_i:05d}.npy"


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
    clip_samples = int(SAMPLE_RATE * CLIP_DURATION)

    # for year in [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]:
    for year in [2018]:
        print(f"📥 Precomputing year {year}")
        loader = MAESTRODataLoader(DATA_ROOT, year=year)
        for split in ("train", "validation", "test"):
            print(f"📥 Precomputing {split} splits")
            for pair in loader.get_pairs(split):
                n_clips = math.floor(pair["duration"] / CLIP_DURATION)
                for i in range(n_clips):
                    path = cqt_path(pair["audio_path"], i)
                    if path.exists():
                        continue
                    audio, _ = librosa.load(
                        pair["audio_path"],
                        sr=SAMPLE_RATE,
                        mono=True,
                        offset=i * CLIP_DURATION,
                        duration=CLIP_DURATION,
                    )
                    audio = helpers.pad_or_trim(audio, clip_samples)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, preprocessor.compute(audio))

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
