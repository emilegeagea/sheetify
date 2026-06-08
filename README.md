# 🎼 Sheetify

> A deep learning automatic music transcription system.

Sheetify is an end-to-end automatic music transcription system that converts piano audio recordings into MIDI files and printable sheet music. Inspired by Google's **Onsets and Frames** architecture, Sheetify uses deep neural networks to detect note onsets, pitches, and durations from raw audio, enabling accurate transcription of polyphonic piano performances.

The model was trained and evaluated using the **MAESTRO dataset**, a large-scale collection of piano performances with precisely aligned audio recordings and MIDI annotations, providing high-quality ground truth data for automatic music transcription.

---

## Overview

Automatic Music Transcription (AMT) is the task of converting audio recordings into symbolic musical representations. Sheetify tackles this challenge by leveraging an onset-aware transcription pipeline that first identifies note beginnings and then predicts note activations over time. This approach significantly improves note-level accuracy compared to traditional frame-based methods.

The resulting note predictions are converted into MIDI format and rendered as standard sheet music, creating a complete audio-to-score workflow.

---

## Features

- 🎹 Piano audio transcription
- 🎵 Audio-to-MIDI conversion
- 🎼 Automatic sheet music generation
- 🧠 Deep learning-based note detection
- 📈 Polyphonic music transcription
- 🔄 End-to-end audio → MIDI → sheet music pipeline

---

## How It Works

### 1. Audio Processing
Input audio is converted into a log-mel spectrogram representation suitable for neural network processing.

### 2. Note Transcription
A deep neural network predicts:

- **Onsets** — when notes begin
- **Frames** — which notes remain active over time

By conditioning frame predictions on onset detections, the model produces cleaner and more accurate transcriptions.

### 3. MIDI Generation
Detected notes are translated into MIDI events, preserving timing and pitch information.

### 4. Sheet Music Rendering
The generated MIDI file is converted into standard musical notation that can be viewed, edited, or printed.

---

## Motivation

Transcribing music manually is time-consuming and requires significant musical expertise. Existing transcription methods often struggle with overlapping notes, harmonic complexity, and note duration estimation.

Sheetify addresses these challenges by implementing an onset-aware architecture inspired by the **Onsets and Frames** paper, which demonstrated that accurate onset detection is critical for high-quality piano transcription. By combining machine learning with symbolic music representation, Sheetify makes it easier to digitize performances and generate readable sheet music automatically.

---

## Tech Stack

- Python
- TensorFlow / PyTorch
- Librosa
- NumPy
- MIDI Processing Libraries
- Music Notation Rendering Tools

---

## Applications

- Music education
- Performance transcription
- Music analysis
- Digital archiving
- Composition and arrangement workflows

---

## Future Improvements

- Multi-instrument transcription
- Real-time transcription support
- Improved note velocity prediction
- Enhanced sheet music formatting
- Larger and more diverse training datasets

---

## Team

Sheetify was developed as a collaborative machine learning and music information retrieval project exploring modern deep-learning approaches to automatic music transcription.

---

## References

Hawthorne, C., Elsen, E., Song, J., Roberts, A., Simon, I., Raffel, C., Engel, J., Oore, S., & Eck, D. (2018).

**Onsets and Frames: Dual-Objective Piano Transcription**

Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR).
