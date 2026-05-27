
def build_tf_dataset(
    maestro_root: str,
    split: str = "train",
    batch_size: int = 16,
    augment: bool = False,
    shuffle_buffer: int = 200,
    prefetch: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """
    Full pipeline: MAESTRO files → batched tf.data.Dataset of (CQT, piano_roll).

    Args:
        maestro_root  : path to maestro-v3.0.0/ directory
        split         : 'train' | 'validation' | 'test'
        batch_size    : number of clips per batch
        augment       : apply augmentations (train only recommended)
        shuffle_buffer: number of samples to shuffle

    Returns:
        tf.data.Dataset yielding:
            cqt        : float32 tensor (batch, N_BINS, time_frames, 1)
            piano_roll : float32 tensor (batch, 88, piano_roll_frames)
    """
    loader = MAESTRODataLoader(maestro_root)
    preprocessor = CQTPreprocessor()
    augmentor = Augmentor() if augment else None

    pairs = loader.get_pairs(split)

    def generator():
        for pair in pairs:
            try:
                audio, roll = loader.load_pair(pair)
                if augmentor:
                    audio, roll = augmentor(audio, roll)
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
