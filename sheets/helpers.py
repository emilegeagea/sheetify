import numpy as np
import matplotlib.pyplot as plt
import pretty_midi

from typing import Tuple


def plot_pianoroll_in_vs_out(
    ground_truth: np.ndarray,
    prediction: np.ndarray
    ) -> None:
    # TODO: Add switch for having time in frames vs seconds
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    axes[0].set_title("Piano Roll (ground truth)")
    axes[0].imshow(ground_truth, aspect="auto", origin="lower", cmap="Blues")
    axes[0].set_xlabel("Time (frames)")
    axes[0].set_ylabel("Piano key (0=A0, 87=C8)")

    axes[1].set_title("Model output")
    axes[1].imshow(prediction, aspect="auto", origin="lower", cmap="Blues")
    axes[1].set_xlabel("Time (frames)")
    axes[1].set_ylabel("Piano key (0=A0, 87=C8)")

    plt.tight_layout()
    plt.show()

def plot_pianoroll(piano_roll: np.ndarray) -> None:
    plt.figure(figsize=(15, 4))
    plt.imshow(piano_roll, aspect="auto", origin="lower", cmap="Blues")

    plt.title("Piano Roll (ground truth)")
    plt.xlabel("Time (frames)")
    plt.ylabel("Piano key (0=A0, 87=C8)")

    plt.show()


def _pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))

def _pad_or_trim_2d(arr: np.ndarray, length: int, axis: int = 1) -> np.ndarray:
    current = arr.shape[axis]
    if current >= length:
        return np.take(arr, range(length), axis=axis)
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, length - current)
    return np.pad(arr, pad_width)


def numpy_to_midi_object(
    piano_roll_88,
    fs: int = 50,
    threshold: float = 0.5
    ) -> pretty_midi.PrettyMIDI:
    """
    Converts an (88, T) piano roll NumPy array into an in-memory PrettyMIDI object.
    """
    # 1. Transpose to (Time, 88) to iterate through time steps row-by-row
    piano_roll = piano_roll_88.T

    # 2. Initialize PrettyMIDI objects
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # 0 = Acoustic Grand Piano

    # Pad the matrix with an extra empty step at the end to catch final note-offs
    padded_roll = np.pad(piano_roll, ((0, 1), (0, 0)), 'constant')
    active_notes = {}

    # 3. Parse time steps
    for step in range(padded_roll.shape[0]):
        current_time = step / fs

        for idx in range(88):
            midi_pitch = idx + 21  # Offset index by 21 (Key 0 maps to A0 / MIDI 21)
            activation_value = padded_roll[step, idx]
            is_note_active = activation_value > threshold

            # Case A: Note just turned on
            if is_note_active and midi_pitch not in active_notes:
                velocity = int(activation_value * 127) if activation_value <= 1.0 else int(activation_value)
                velocity = max(min(velocity, 127), 1)  # Clamp between 1 and 127
                active_notes[midi_pitch] = (current_time, velocity)

            # Case B: Note just turned off
            elif not is_note_active and midi_pitch in active_notes:
                start_time, note_velocity = active_notes.pop(midi_pitch)

                note = pretty_midi.Note(
                    velocity=note_velocity,
                    pitch=midi_pitch,
                    start=start_time,
                    end=current_time
                )
                instrument.notes.append(note)

    pm.instruments.append(instrument)
    return pm


def midi_object_to_playable(
    pm: pretty_midi.PrettyMIDI,
    sample_rate: int = 44100,
    ) -> Tuple[np.ndarray, int]:
    """
    Synthesizes audio directly from an in-memory PrettyMIDI object.
    """
    total_time = pm.get_end_time()
    total_samples = int(total_time * sample_rate)
    audio_data = np.zeros(total_samples)

    for instrument in pm.instruments:
        for note in instrument.notes:
            start_sample = int(note.start * sample_rate)
            end_sample = int(note.end * sample_rate)

            if end_sample <= start_sample:
                continue

            # Convert MIDI pitch to frequency (Hz)
            frequency = pretty_midi.note_number_to_hz(note.pitch)

            # Generate the sound wave for this note's duration
            duration = (end_sample - start_sample) / sample_rate
            t = np.linspace(0, duration, end_sample - start_sample, endpoint=False)

            # Smooth out the volume fade so notes don't aggressively "click" when ending
            fade_out = np.linspace(1.0, 0.0, len(t)) ** 0.5
            note_wave = np.sin(2 * np.pi * frequency * t) * 0.1 * fade_out

            # Mix this note into the master song audio track
            audio_data[start_sample:end_sample] += note_wave

    # Normalize audio levels to prevent clipping/distortion
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    print('''
          Usage:
          from IPython.display import Audio
          Audio(pm_array[0], rate=pm_array[1])
          ''')

    return audio_data, sample_rate
