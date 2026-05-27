import librosa

SAMPLE_RATE     = 16_000   # Hz
CLIP_DURATION   = 10.0     # seconds
HOP_LENGTH      = 512      # ~32ms at 16kHz
N_BINS          = 84       # 7 octaves × 12 semitones (full piano range)
BINS_PER_OCTAVE = 12
N_MELS          = 84       # changed from 128 → must match model input (84, 313, 1)
FMIN = librosa.note_to_hz("A0")  # 27.5 Hz — lowest piano key
PIANO_ROLL_FS   = 100      # frames per second — matches CQT pipeline
PIANO_MIN_PITCH = 21       # MIDI A0
PIANO_MAX_PITCH = 108      # MIDI C8
N_PIANO_KEYS    = PIANO_MAX_PITCH - PIANO_MIN_PITCH + 1  # 88
