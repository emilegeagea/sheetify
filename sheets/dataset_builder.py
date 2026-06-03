import os
import tensorflow as tf
import math

from sheets.augmentor import Augmentor
from sheets.preprocessors import Preprocessor
from sheets.dataloader import MAESTRODataLoader

from sheets.target_builder import piano_roll_to_onset_frame

from sheets.constants import *


def _resolve_num_shards(num_parallel_calls: int | None) -> int:
    """Shard count for interleave; 1 keeps the original single-generator path."""
    if num_parallel_calls == 1:
        return 1
    if num_parallel_calls is None or num_parallel_calls == tf.data.AUTOTUNE:
        return os.cpu_count() or 8
    return int(num_parallel_calls)


def _interleave_from_generators(
    num_shards: int,
    make_generator,
    output_signature,
    num_parallel_calls: int | None,
) -> tf.data.Dataset:
    """
    Run `num_shards` generator instances in parallel via Dataset.interleave.
    `make_generator(shard_index, num_shards)` must return the generator callable.
    """
    if num_parallel_calls is None:
        num_parallel_calls = tf.data.AUTOTUNE

    # Shard index in interleave is a SymbolicTensor; build datasets with Python ints.
    shard_datasets = [
        tf.data.Dataset.from_generator(
            make_generator(i, num_shards),
            output_signature=output_signature,
        )
        for i in range(num_shards)
    ]

    @tf.autograph.experimental.do_not_convert
    def dataset_for_shard(shard_index):
        shard_index = tf.cast(shard_index, tf.int32)
        return tf.switch_case(
            shard_index,
            branch_fns=[lambda ds=ds: ds for ds in shard_datasets],
        )

    return tf.data.Dataset.range(num_shards).interleave(
        dataset_for_shard,
        cycle_length=num_shards,
        num_parallel_calls=num_parallel_calls,
        deterministic=False,
    )


def build_tf_dataset(
    dataloader: MAESTRODataLoader,
    preprocessor: Preprocessor,
    augementor: Augmentor | None = None,
    split: str = "train",
    batch_size: int = 16,
    augment: bool = False,
    shuffle_buffer: int = 10000,
    prefetch: int = tf.data.AUTOTUNE,
    num_parallel_calls: int | None = None,
) -> tf.data.Dataset:
    """
    Full pipeline: MAESTRO files → batched tf.data.Dataset of (CQT, piano_roll).

    Args:
        split         : 'train' | 'validation' | 'test'
        batch_size    : number of clips per batch
        augment       : apply augmentations (train only recommended)
        shuffle_buffer: number of samples to shuffle
        num_parallel_calls: interleave parallelism (default AUTOTUNE); use 1 for original single generator

    Returns:
        tf.data.Dataset yielding:
            cqt        : float32 tensor (batch, N_BINS, time_frames, 1)
            piano_roll : float32 tensor (batch, 88, piano_roll_frames)
    """
    pairs = dataloader.get_pairs(split)
    num_shards = _resolve_num_shards(num_parallel_calls)

    @tf.autograph.experimental.do_not_convert
    def make_generator(shard_index: int, num_shards: int):
        @tf.autograph.experimental.do_not_convert
        def generator():
            for pair_index, pair in enumerate(pairs):
                if pair_index % num_shards != shard_index:
                    continue
                num_splits = math.floor(pair["duration"] / CLIP_DURATION)
                split_starts = [CLIP_DURATION * idx for idx in range(num_splits)]
                for split_start in split_starts:
                    try:
                        # audio, roll = dataloader.load_pair(pair, split_start)
                        # preproc = preprocessor.compute(audio)
                        roll = dataloader.load_roll(pair, split_start)
                        preproc = dataloader.load_CQT(pair, split_start, preprocessor=preprocessor)
                        yield preproc, roll
                    except Exception as e:
                        print(
                            f"[Warning] Skipping {pair['audio_path']} at {split_start} seconds: {e}"
                        )
                        continue

        return generator

    # Infer output shapes
    n_frames = int(CLIP_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1
    roll_frames = int(CLIP_DURATION * PIANO_ROLL_FS)

    output_signature = (
        tf.TensorSpec(shape=(N_BINS, n_frames, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(N_PIANO_KEYS, roll_frames), dtype=tf.float32),
    )

    if num_shards == 1:
        dataset = tf.data.Dataset.from_generator(
            make_generator(0, 1),
            output_signature=output_signature,
        )
    else:
        dataset = _interleave_from_generators(
            num_shards,
            make_generator,
            output_signature,
            num_parallel_calls,
        )

    cache_dir = "./.sheetify_cache"
    os.makedirs(cache_dir, exist_ok=True)
    preprocessor_name = type(preprocessor).__name__.replace("Preprocessor", "")
    dataset = dataset.cache(cache_dir + f"/cache_{split}_{preprocessor_name}")   # cache on disk after first epoch

    # if split == "train":
        # dataset = dataset.shuffle(shuffle_buffer)

    dataset = (
        dataset
        .batch(
            batch_size,
            drop_remainder=False # TODO: Investigate whether this should be True or False
                                 # When True, the model.fit raises a math.domain exception on a logarithm
                                 # The cause of this is unknown
        ).prefetch(prefetch)
    )

    return dataset


def build_onf_dataset(
    dataloader: MAESTRODataLoader,
    preprocessor: Preprocessor,
    split: str = "train",
    batch_size: int = 16,
    shuffle_buffer: int = 200,
    prefetch: int = tf.data.AUTOTUNE,
    num_parallel_calls: int | None = None,
) -> tf.data.Dataset:
    pairs = dataloader.get_pairs(split)
    num_shards = _resolve_num_shards(num_parallel_calls)

    @tf.autograph.experimental.do_not_convert
    def make_generator(shard_index: int, num_shards: int):
        @tf.autograph.experimental.do_not_convert
        def generator():
            for pair_index, pair in enumerate(pairs):
                if pair_index % num_shards != shard_index:
                    continue
                num_splits = math.floor(pair["duration"] / CLIP_DURATION)
                split_starts = [CLIP_DURATION * idx for idx in range(num_splits)]
                for split_start in split_starts:
                    try:
                        # audio, roll = dataloader.load_pair(pair, split_start)
                        # preproc = preprocessor.compute(audio)
                        roll = dataloader.load_roll(pair, split_start)
                        onf_rolls = piano_roll_to_onset_frame(roll)

                        preproc = dataloader.load_CQT(pair, split_start, preprocessor=preprocessor)
                        yield preproc, (onf_rolls[0], onf_rolls[1])
                    except Exception as e:
                        print(
                            f"[Warning] Skipping {pair['audio_path']} at {split_start} seconds: {e}"
                        )
                        continue

        return generator

    n_frames = int(CLIP_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1
    roll_frames = int(CLIP_DURATION * PIANO_ROLL_FS)

    output_signature = (
        tf.TensorSpec(shape=(N_BINS, n_frames, 1), dtype=tf.float32, name="feat_spectrogram"),
        (
            tf.TensorSpec(shape=(N_PIANO_KEYS, roll_frames), dtype=tf.float32, name="target_onset_roll"),
            tf.TensorSpec(shape=(N_PIANO_KEYS, roll_frames), dtype=tf.float32, name="target_frame_roll"),
        ),
    )

    if num_shards == 1:
        dataset = tf.data.Dataset.from_generator(
            make_generator(0, 1),
            output_signature=output_signature,
        )
    else:
        dataset = _interleave_from_generators(
            num_shards,
            make_generator,
            output_signature,
            num_parallel_calls,
        )

    # if split == "train":
        # dataset = dataset.shuffle(shuffle_buffer)

    dataset = (
        dataset
        .batch(
            batch_size,
            drop_remainder=False # TODO: Investigate whether this should be True or False
                                 # When True, the model.fit raises a math.domain exception on a logarithm
                                 # The cause of this is unknown
        ).prefetch(prefetch)
    )

    return dataset
