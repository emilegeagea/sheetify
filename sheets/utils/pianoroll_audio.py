import numpy as np
import pretty_midi

from typing import Tuple

def numpy_to_midi_object(
    piano_roll_88,
    fs: int = 100,
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

def numpy_to_playable(piano_roll):
    mo = numpy_to_midi_object(piano_roll)
    pm = midi_object_to_playable(mo)
    return pm