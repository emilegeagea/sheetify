import numpy as np
import pretty_midi

from typing import Tuple

def numpy_to_midi_object(piano_roll_88, fs: int = 100, threshold: float = 0.5) -> pretty_midi.PrettyMIDI:
    """Converts an (88, T) piano roll NumPy array into an in-memory PrettyMIDI object."""
    piano_roll = piano_roll_88.T
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # 0 = Acoustic Grand Piano

    padded_roll = np.pad(piano_roll, ((0, 1), (0, 0)), 'constant')
    active_notes = {}

    for step in range(padded_roll.shape[0]):
        current_time = step / fs

        for idx in range(88):
            midi_pitch = idx + 21  
            activation_value = padded_roll[step, idx]
            is_note_active = activation_value > threshold

            if is_note_active and midi_pitch not in active_notes:
                velocity = int(activation_value * 127) if activation_value <= 1.0 else int(activation_value)
                velocity = max(min(velocity, 127), 1)  
                active_notes[midi_pitch] = (current_time, velocity)

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


def midi_object_to_playable(pm: pretty_midi.PrettyMIDI, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Synthesizes rich, acoustic-sounding piano notes directly from a PrettyMIDI object
    using multi-harmonic additive synthesis and an exponential string decay envelope.
    """
    total_time = pm.get_end_time() + 0.5  # Added padding for note tail ringing
    total_samples = int(total_time * sample_rate)
    audio_data = np.zeros(total_samples)

    for instrument in pm.instruments:
        for note in instrument.notes:
            start_sample = int(note.start * sample_rate)
            # Extend end sample slightly to allow the note to fade out naturally after release
            end_sample = int((note.end + 0.3) * sample_rate)

            if start_sample >= total_samples:
                continue
            if end_sample > total_samples:
                end_sample = total_samples

            duration_samples = end_sample - start_sample
            if duration_samples <= 0:
                continue

            f0 = pretty_midi.note_number_to_hz(note.pitch)
            t = np.linspace(0, duration_samples / sample_rate, duration_samples, endpoint=False)

            # 1. Multi-Harmonic Additive Synthesis (Gives the piano timbre its depth)
            # Structure: Fundamental frequency + weaker, higher-frequency string reflections
            wave = (
                1.00 * np.sin(2 * np.pi * f0 * t) +        # Fundamental Tone
                0.45 * np.sin(2 * np.pi * (2 * f0) * t) +  # 2nd Harmonic
                0.25 * np.sin(2 * np.pi * (3 * f0) * t) +  # 3rd Harmonic
                0.12 * np.sin(2 * np.pi * (4 * f0) * t)    # 4th Harmonic
            )

            # 2. String Resonance Decay Modelling (ADSR)
            # Real piano strings strike instantaneously, decay exponentially, and damp at release
            note_duration = note.end - note.start
            
            # Fast attack (0.005s) to capture the acoustic hammer strike crackle
            attack_samples = int(0.005 * sample_rate)
            envelope = np.ones(duration_samples)
            
            if attack_samples < duration_samples:
                envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
            
            # Exponential decay factor simulating string resonance
            decay_factor = 2.5 if f0 < 250 else 4.0  # Bass strings ring longer than high treble
            decay_env = np.exp(-decay_factor * t)
            envelope = envelope * decay_env

            # Damper pedal simulation: Rapidly damp vibrations after the note key release frame
            release_start_idx = int(note_duration * sample_rate)
            if release_start_idx < duration_samples:
                release_samples = duration_samples - release_start_idx
                # Apply structural damping linear window down to zero volume
                damp_window = np.linspace(1.0, 0.0, release_samples)
                envelope[release_start_idx:] *= damp_window

            # Normalize note velocity scale relative to MIDI parameters
            amplitude = (note.velocity / 127.0) * 0.15
            note_wave = wave * envelope * amplitude

            # Mix note segment into global audio space array
            audio_data[start_sample:end_sample] += note_wave

    # 3. Dynamic Range Compressor / Peak Normalization to protect against digital clipping
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val

    return audio_data, sample_rate


def numpy_to_playable(piano_roll):
    mo = numpy_to_midi_object(piano_roll)
    pm = midi_object_to_playable(mo)
    return pm