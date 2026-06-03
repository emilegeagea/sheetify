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

# Internal project modules
from sheets.main import run_prediction, load_ml_model
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

def run_prediction(uploaded_file, model):
    preprocessed_data = preprocess(uploaded_file)
    pred_output = predict(preprocessed_data)
    post_pros = postprocess(pred_output)
    return post_pros

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
path = "/home/pom/code/emilegeagea/sheetify/2004-allsplits-onf.keras" # make sure maybe we need to use SYS library to access the path correctly since this is not going to be the same path on Docker container 

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
                        label="📄 Download Sheet Music PDF",
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