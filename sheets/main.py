import numpy as np
import pandas as pd

from colorama import Fore, Style

from sheets.params import *

from sheets.dataloader import MAESTRODataLoader
from sheets.preprocessors import CQTPreprocessor
import sheets.basicmodel as basicmodel
from sheets.dataset_builder import build_tf_dataset

from sheets.utils.registry import mlflow_run, save_model, save_results

import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

@mlflow_run
def train(
    limit=None,
    batch_size=8,
    patience=5,
    epochs=20
):
    dataloader = MAESTRODataLoader(limit=limit)
    preprocessor = CQTPreprocessor()

    train_ds = build_tf_dataset(
        dataloader, preprocessor, batch_size=batch_size, split='train')
    val_ds = build_tf_dataset(
        dataloader, preprocessor, batch_size=batch_size, split='validation')

    model = basicmodel.initialize_model()
    model = basicmodel.compile_model(model)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        filepath=f'./models/{timestamp}.weights.h5',
        save_weights_only=True,
        # monitor='val_fbeta',
        mode='max',
        save_best_only=True
    )

    es = EarlyStopping(
        patience=patience
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    # Save model locally
    # model_path = os.path.join("./models", f"{timestamp}.h5")
    # model.save(model_path)


    fbeta = np.min(history.history['fbeta'])

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
