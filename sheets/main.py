import numpy as np
import pandas as pd

from colorama import Fore, Style

from sheets.params import *

from sheets.dataloader import MAESTRODataLoader
from sheets.preprocessors import CQTPreprocessor
import sheets.basicmodel as basicmodel
import sheets.onf_model as onf_model
from sheets.dataset_builder import build_tf_dataset, build_onf_dataset

from sheets.utils.registry import mlflow_run, save_model, save_results

import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

@mlflow_run
def train(
    model_type='basic',
    limit=None,
    year=YEAR_LIMIT,
    batch_size=32,
    patience=10,
    epochs=200
):
    dataloader = MAESTRODataLoader(limit=limit, year=year)
    preprocessor = CQTPreprocessor()

    if model_type == 'basic':
        print('📌 Training Basic model type.')
        train_ds = build_tf_dataset(
            dataloader, preprocessor, batch_size=batch_size, split='train')
        val_ds = build_tf_dataset(
            dataloader, preprocessor, batch_size=batch_size, split='validation')

        model = basicmodel.initialize_model()
        model = basicmodel.compile_model(model)
    elif model_type == 'onf':
        print('📌 Training ONF model type.')
        train_ds = build_onf_dataset(
            dataloader, preprocessor, batch_size=batch_size, split='train')
        val_ds = build_onf_dataset(
            dataloader, preprocessor, batch_size=batch_size, split='validation')

        model = onf_model.initialize_model()
        model = onf_model.compile_model(model)
    else:
        print('❌ No model type to train selected. Exiting.')
        raise SystemExit


    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = f'./models/{timestamp}.keras'
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True
    )

    es = EarlyStopping(
        patience=patience
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[es, checkpoint_callback],
        verbose=2
    )

    # Load the best weights
    model = tf.keras.models.load_model(
        checkpoint_filepath,
        safe_mode=False
        )

    # Save model locally
    # model_path = os.path.join("./models", f"{timestamp}.keras")
    # model.save(model_path)


    try:
        fbeta = np.min(history.history['fbeta'])
    except Exception:
        fbeta = 0

    params = dict(
        context="train",
        limit=limit,
        dataset_builder_batch_size=batch_size
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(fbeta=fbeta))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)


    print("✅ train() done \n")
