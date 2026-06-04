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
import pretty_midi
from streamlit_pdf_viewer import pdf_viewer
from typing import Tuple

# Internal project modules
from sheets.main import run_prediction, load_ml_model
from sheets.midi_to_note import midi_to_note
from sheets.preprocessors import CQTPreprocessor
from sheets.utils.plotting import plot_pianoroll, plot_pianoroll_overlay
from sheets.utils.pianoroll_audio import midi_object_to_playable

# ─── NATIVE PAGE CONFIGURATION & MODERN STYLING ──────────────────────────────
st.set_page_config(
    page_title="Sheetify",
    page_icon="🎹",
    layout="wide"
)

# Modern Slate & Cyber-Mint CSS Injection with optimized tight container margins
st.markdown("""
<style>
    /* Main Background Base */
    .stApp {
        background-color: #F8FAFC;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        padding-bottom: 0px !important;
    }
    
    /* Modern Card containers styling with minimized inner padding for a smaller footprint */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 0px !important;
    }
    
    /* Primary Call to Action Button: Cyber Mint Style */
    button[kind="primary"] {
        margin-top: 5px !important;
        background-color: #10B981 !important;
        border-color: #10B981 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: background-color 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background-color: #059669 !important;
        border-color: #059669 !important;
    }

    /* Secondary / Waiting State Button Style */
    button[kind="secondary"] {
        margin-top: 5px !important;
        background-color: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        color: #64748B !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── PREPROCESSING & MODEL FUNCTIONS ─────────────────────────────────────────

def preprocess(uploaded_file):
    audio_data, sampling_rate = librosa.load(
        uploaded_file,
        sr=16_000,
        duration=10,
        offset=0,
        mono=True,
    )
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
    pred_array = np.asarray(prediction_output[1]).reshape(88, 1000)
    scaled_preds = (pred_array * 255).astype(np.uint8)
    
    _, binary_piano_roll = cv2.threshold(
        scaled_preds, 100, 255, cv2.THRESH_BINARY
    )
    return (binary_piano_roll / 255).astype(np.int32)

def clean_piano_roll(piano_roll, fs=100, min_duration_ms=40, max_gap_ms=40):
    min_frames = int(min_duration_ms / 1000 * fs)
    max_gap_frames = int(max_gap_ms / 1000 * fs)
    cleaned = np.zeros_like(piano_roll, dtype=np.float32)

    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    for key in range(88):
        row = piano_roll[key].copy()
        if not np.any(row):
            continue

        changes = np.diff(row, prepend=0, append=0)
        onsets  = np.where(changes == 1)[0]
        offsets = np.where(changes == -1)[0]

        for i in range(len(onsets) - 1):
            if onsets[i+1] - offsets[i] <= max_gap_frames:
                row[offsets[i]:onsets[i+1]] = 1

        changes = np.diff(row, prepend=0, append=0)
        onsets  = np.where(changes == 1)[0]
        offsets = np.where(changes == -1)[0]

        for start, end in zip(onsets, offsets):
            if (end - start) >= min_frames:
                cleaned[key, start:end] = 1.0
                
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=key + 21,
                    start=start / fs,
                    end=end / fs,
                )
                piano.notes.append(note)

    midi.instruments.append(piano)
    return cleaned, midi

def run_prediction(uploaded_file, model):
    preprocessed_data = preprocess(uploaded_file)
    pred_output = predict(preprocessed_data)
    post_pros = postprocess(pred_output) 
    cleaned_roll, midi_obj = clean_piano_roll(post_pros) 
    return cleaned_roll, midi_obj 


# ─── RENDERING & UI VISUALIZATION FUNCTIONS ──────────────────────────────────

def show_audio(midi_obj):
    audio_data, sample_rate = midi_object_to_playable(midi_obj)
    st.audio(audio_data, sample_rate=sample_rate)
    
def plot_pianoroll(piano_roll: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(15, 3.5))
    ax.imshow(piano_roll, aspect="auto", origin="lower", cmap="Blues")
    
    ax.set_title("Piano Roll Visualization", fontsize=10, pad=5, color='#0F172A', fontweight='bold')
    ax.set_xlabel("Time (frames)", fontsize=8, color='#64748B')
    ax.set_ylabel("Piano key (0=A0, 87=C8)", fontsize=8, color='#64748B')
    ax.tick_params(axis='both', which='major', labelsize=8, colors='#64748B')
    
    st.pyplot(fig)
    plt.close(fig)


# ─── MODEL INITIALIZATION ────────────────────────────────────────────────────

local_path = "/home/pom/code/emilegeagea/sheetify/2004-2006-2008-2009-2011-2013-allsplits-onf.keras"
if os.path.exists(local_path):
    path = local_path
else:
    path = os.path.join(os.path.dirname(__file__), "2004-allsplits-onf.keras")

@st.cache_resource
def get_cached_model(model_path):
    return load_ml_model(model_path)

model = get_cached_model(path)


# ─── FIXED GRID LAYOUT SETUP (50/50 HALF PAGE SPLIT) ─────────────────────────

st.markdown("<h2 style='margin: 0px; padding: 0px; color: #0F172A; font-weight:700;'>🎹 Sheetify &mdash; Audio to Sheet Music</h2>", unsafe_allow_html=True)

# Layout configuration split perfectly into half of the viewport width
left, right = st.columns([1, 1], gap="medium")

# Pre-allocating Left Column layout blocks
with left:
    with st.container(border=True):
        sub_col1, sub_col2 = st.columns([2.5, 1.5], gap="small")
        
        with sub_col1:
            uploaded_file = st.file_uploader(
                "Upload target MP3 file:", 
                type="mp3",
                label_visibility="collapsed"
            )
            
        with sub_col2:
            if uploaded_file is not None:
                predict_btn = st.button('🚀 Transcribe & Generate Score', use_container_width=True, type="primary")
            else:
                st.button('Waiting for MP3 File...', use_container_width=True, disabled=True)
        
        # Status loading area
        status_placeholder = st.empty()
        
    # Pre-allocated structural space for Left Column features
    audio_playback_container = st.empty()
    piano_roll_container = st.empty()

# Pre-allocating Right Column layout blocks
with right:
    with st.container(border=True):
        st.markdown("<div style='color: #334155;'><b>How it works:</b> Upload MP3 &rarr; ML transcribes notes &rarr; Preview & download score.</div>", unsafe_allow_html=True)
        
    # Pre-allocated target container directly below instructions for the PDF Engine outputs
    pdf_container = st.empty()


# ─── INFERENCE RESULTS PIPELINE ──────────────────────────────────────────────

if uploaded_file is not None and 'predict_btn' in locals() and predict_btn:
    
    with status_placeholder.status("Transcribing audio matrix...", expanded=False) as status:
        post_processed_output, optimized_midi = run_prediction(uploaded_file, model)
        score_object = midi_to_note(post_processed_output)
        status.update(label="Complete!", state="complete")
    
    # 1. Populate Left Hand Side Blocks (Visualizations/Playback)
    with audio_playback_container.container(border=True):
        st.markdown("<b style='color: #0F172A;'>Audio Playback</b>", unsafe_allow_html=True)
        show_audio(optimized_midi)
        
    with piano_roll_container.container(border=True):
        plot_pianoroll(post_processed_output)
        
    # 2. Populate Right Hand Side Blocks (PDF Interface Engine taking up half the page)
    with pdf_container.container(border=True):
        try:
            pdf_output_path = score_object.write('lilypond.pdf', fp='sheetify_output')
            pdf_filepath = str(pdf_output_path)
            
            if os.path.exists(pdf_filepath):
                with open(pdf_filepath, "rb") as f:
                    pdf_bytes = f.read()
                
                st.markdown("<b style='color: #0F172A; margin-bottom: 4px; display: block;'>Live Score Preview:</b>", unsafe_allow_html=True)
                
                # Removed static width pixel size so the element naturally occupies half the page layout flexbox
                pdf_viewer(input=pdf_bytes, height=320)
                
                st.divider()
                
                st.download_button(
                    label="📥 Download Sheet Music PDF",
                    data=pdf_bytes,
                    file_name="transcribed_score.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
                
                # Cleanup Operations
                os.remove(pdf_filepath)
                ly_file = pdf_filepath.replace('.pdf', '.ly')
                if os.path.exists(ly_file):
                    os.remove(ly_file)
            else:
                st.error("Engine asset generation mismatch.")
                
        except Exception as e:
            st.error(f"Compilation layout breakdown: {e}")