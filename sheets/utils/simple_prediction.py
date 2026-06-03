import numpy as np
from typing import Any

import sheets.utils.plotting as plotting
from sheets.dataloader import MAESTRODataLoader
from sheets.preprocessors import CQTPreprocessor

from sheets.constants import *

simple_pair = {
    'audio_path': './data/mp3s/midi.mp3',
    'midi_path':  './data/midis/midi.mid',
    'duration': 26
}

complex_pair = {
    'audio_path': './data/mp3s/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.mp3',
    'midi_path': './data/midis/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi',
    'duration': 26
}

def predict_roll(
    model,
    dataloader = MAESTRODataLoader(),
    preprocessor = CQTPreprocessor(),
    pair: dict[str, Any] = simple_pair,
    start_sec: int = 0,
):
    audio, roll = dataloader.load_pair(pair, start_sec=start_sec)
    preproc = preprocessor.compute(audio)

    preproc_input = np.expand_dims(preproc, axis=0)    # (1, 84, 313, 1)

    pred = model.predict(preproc_input, verbose=0)
    if isinstance(pred, list):
        pred = pred[1]

    pred_reshaped = np.reshape(pred, newshape=(88, 1000))
    roll_reshaped = np.reshape(roll, newshape=(88, 1000))

    return pred_reshaped, roll_reshaped


def predict_pair(model, pair, **kwargs):
    pred_reshaped, roll_reshaped = predict_roll(model, pair=pair, **kwargs)
    plotting.plot_pianoroll_in_vs_out(
        roll_reshaped,
        pred_reshaped)


def predict_simple(model, **kwargs):
    print('Running model prediction on simple scale')
    predict_pair(model, pair=simple_pair, **kwargs)

def predict_simple_triples(model, **kwargs):
    print('Running model prediction on simple scale with triple notes')
    predict_pair(model, pair=simple_pair, start_sec=13, **kwargs)

def predict_complex(model, **kwargs):
    print('Running model prediction on Chamber2_09_R3_2018')
    predict_pair(model, pair=complex_pair, **kwargs)
