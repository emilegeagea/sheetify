SAMPLE_RATE     = 16_000   # resampled to match CQT pipeline
CLIP_DURATION   = 10.0     # seconds — matches model input
HOP_LENGTH      = 512      # matches CQT pipeline
N_MELS          = 84       # changed from 128 → must match model input (84, 313, 1)
PIANO_ROLL_FS   = 100      # frames per second — matches CQT pipeline
PIANO_MIN_PITCH = 21       # MIDI A0
PIANO_MAX_PITCH = 108      # MIDI C8
N_PIANO_KEYS    = PIANO_MAX_PITCH - PIANO_MIN_PITCH + 1  # 88
