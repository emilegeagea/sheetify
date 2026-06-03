import streamlit as st
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
import librosa
import numpy as np
import keras.src.layers.layer as _layer_module
import os
import pretty_midi  # Fixed: Required for clean_piano_roll execution

# Internal project modules
# from sheets.main import run_prediction, load_ml_model
from sheets.midi_to_note import midi_to_note
from sheets.preprocessors import CQTPreprocessor
from sheets.utils.plotting import plot_pianoroll, plot_pianoroll_overlay
from sheets.utils.pianoroll_audio import numpy_to_playable

'''
# Sheetify
'''

st.markdown('''
Turning mp3 into sheet music.
''')

# --- PREPROCESSING & MODEL FUNCTIONS ---

def preprocess(uploaded_file):
    audio_data, sampling_rate = librosa.load(uploaded_file, sr=16_000, duration=10)
    preprocessor = CQTPreprocessor()
    features = preprocessor.compute(audio_data)
    preprocessed_audio_data = np.expand_dims(features, axis=0)
    return preprocessed_audio_data

def load_ml_model(path):
    _orig_layer_init = _layer_module.Layer.__init__

    def _compat_layer_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        _orig_layer_init(self, *args, **kwargs)

    _layer_module.Layer.__init__ = _compat_layer_init

    model = tf.keras.models.load_model(path)
    return model

@tf.function(reduce_retracing=True)
def predict(preprocessed_audio):
    predictions = model(preprocessed_audio, training=False)
    return predictions

def postprocess(prediction_output):
    pred_reshaped = np.reshape(prediction_output[1], shape=(88, 1000))
    pred_array = np.array(pred_reshaped)
    pred_array = np.squeeze(pred_array)

    scaled_preds = (pred_array * 255).astype(np.uint8)
    thresh_val, binary_piano_roll = cv2.threshold(
        scaled_preds, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary_piano_roll = (binary_piano_roll / 255).astype(np.int32)
    return binary_piano_roll

def clean_piano_roll(piano_roll, fs=100, min_duration_ms=150, max_gap_ms=100):
    min_frames = int(min_duration_ms / 1000 * fs)
    max_gap_frames = int(max_gap_ms / 1000 * fs)
    cleaned = np.zeros_like(piano_roll)

    # ── Step 1: Find main pitch range ───────────────
    activity = piano_roll.sum(axis=1)
    main_key = np.argmax(activity)
    pitch_tolerance = 15
    pitch_min = max(0,  main_key - pitch_tolerance)
    pitch_max = min(87, main_key + pitch_tolerance)

    for key in range(pitch_min, pitch_max + 1):
        row = piano_roll[key].copy()

        # ── Step 2: Fill short gaps ──────────────────
        changes = np.diff(row, prepend=0, append=0)
        onsets  = np.where(changes == 1)[0]
        offsets = np.where(changes == -1)[0]

        for i in range(len(onsets) - 1):
            if onsets[i+1] - offsets[i] <= max_gap_frames:
                row[offsets[i]:onsets[i+1]] = 1.0

        # ── Step 3: Remove short notes ───────────────
        changes = np.diff(row, prepend=0, append=0)
        onsets  = np.where(changes == 1)[0]
        offsets = np.where(changes == -1)[0]

        for start, end in zip(onsets, offsets):
            if (end - start) >= min_frames:
                cleaned[key, start:end] = 1.0

    # ── Step 4: Piano roll → MIDI ───────────────────
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    for key in range(88):
        midi_pitch = key + 21
        row = cleaned[key]
        changes = np.diff(row, prepend=0, append=0)
        onsets  = np.where(changes == 1)[0]
        offsets = np.where(changes == -1)[0]

        for start, end in zip(onsets, offsets):
            note = pretty_midi.Note(
                velocity=80,
                pitch=midi_pitch,
                start=start / fs,
                end=end / fs,
            )
            piano.notes.append(note)

    midi.instruments.append(piano)

    # ── Step 5: Note-based cleaning (velocity + duration) ──
    for instrument in midi.instruments:
        instrument.notes = [
            note for note in instrument.notes
            if (note.end - note.start) > 0.04  # remove notes < 40ms
            and note.velocity > 15              # remove very quiet notes
        ]

    # ── Step 6: MIDI → cleaned piano roll ───────────
    roll = midi.get_piano_roll(fs=fs)
    roll = roll[21:109, :]
    roll = (roll > 0).astype(np.float32)

    # Match original length
    target = piano_roll.shape[1]
    if roll.shape[1] < target:
        roll = np.pad(roll, ((0,0), (0, target - roll.shape[1])))
    else:
        roll = roll[:, :target]

    return roll, midi

def run_prediction(uploaded_file, model):
    preprocessed_data = preprocess(uploaded_file)
    pred_output = predict(preprocessed_data)
    post_pros = postprocess(pred_output)

    # Fixed: Unpacking tuple correctly so we only return the target matrix to components
    cleaned_roll, midi_obj = clean_piano_roll(post_pros)
    return cleaned_roll

# --- RENDERING & UI VISUALIZATION FUNCTIONS ---

def show_audio(binary_roll3):
    pm_array = numpy_to_playable(binary_roll3)
    audio_data = pm_array[0]
    sample_rate = pm_array[1]
    st.audio(audio_data, sample_rate=sample_rate)

def plot_pianoroll(piano_roll: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.imshow(piano_roll, aspect="auto", origin="lower", cmap="Blues")
    ax.set_title("Piano Roll")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Piano key (0=A0, 87=C8)")
    st.pyplot(fig)
    plt.close(fig)

# --- MAIN APP LOGIC LOOP ---

uploaded_file = st.file_uploader("Choose an mp3 file", type="mp3")

# Fixed: Cloud-Run/Docker safe dynamic path lookup fallback
# local_path = "/home/pom/code/emilegeagea/sheetify/2004-allsplits-onf.keras"
local_path = "/code/models/2004-allsplits-onf.keras" # make sure maybe we need to use SYS library to access the path correctly since this is not going to be the same path on Docker container
if os.path.exists(local_path):
    path = local_path
else:
    # Looks for the file inside your container directory relative to app.py
    path = os.path.join(os.path.dirname(__file__), "2004-allsplits-onf.keras")

@st.cache_resource
def get_cached_model(model_path):
    return load_ml_model(model_path)

model = get_cached_model(path)

if uploaded_file is not None:
    if st.button('Predict'):
        st.info("Processing your audio... Please wait.")

        # 1. Run Machine Learning Pipeline
        post_processed_output = run_prediction(uploaded_file, model)

        # 2. Render Piano Roll Graph
        st.subheader("Piano Roll Visualization")
        with st.spinner("Rendering interface data..."):
            plot_pianoroll(post_processed_output)
        st.success('Inference Complete!')

        # 3. Render Synced Audio Synthesis Player
        st.subheader("Synthesized Audio Playback")
        show_audio(post_processed_output)

        # 4. Generate Music21 Score Object Matrix (In-Memory Only)
        score_object = midi_to_note(post_processed_output)

        # 5. Compile Sheet Music PDF Document
        st.subheader("Download Sheet Music PDF")

        with st.spinner("Compiling publication-quality PDF via LilyPond..."):
            try:
                # music21 executes the headless LilyPond backend to compile a raw vector PDF
                pdf_output_path = score_object.write('lilypond.pdf', fp='sheetify_output')

                # Convert Path object safely into a standard text string
                pdf_filepath = str(pdf_output_path)

                # Check if file compilation succeeded
                if os.path.exists(pdf_filepath):
                    with open(pdf_filepath, "rb") as f:
                        pdf_bytes = f.read()

                    # Present a secure download button for the generated PDF binary
                    st.success("PDF sheet music generated successfully!")
                    st.download_button(
                        label=":page_facing_up: Download Sheet Music PDF",
                        data=pdf_bytes,
                        file_name="transcribed_score.pdf",
                        mime="application/pdf"
                    )

                    # Clean up the container's disk space immediately after loading bytes
                    os.remove(pdf_filepath)

                    # LilyPond leaves a background source .ly file behind; clear it up safely
                    ly_file = pdf_filepath.replace('.pdf', '.ly')
                    if os.path.exists(ly_file):
                        os.remove(ly_file)
                else:
                    st.error("Engine failed to output a valid PDF asset.")

            except Exception as e:
                st.error(f"Failed to compile PDF sheet music: {e}")
                st.info("Ensure LilyPond is installed in your system path environment.")
