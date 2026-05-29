import music21
from showscore import show

def midi_to_note2(input_prettymidiobject, time_signature='4/4'):
    """
    Converts a pretty_midi object into a clean, human-readable 
    two-handed (Grand Staff) piano score within a Jupyter notebook.
    
    It applies aggressive quantization, key detection, and duration 
    rounding to eliminate common visual clutter caused by raw human MIDI data.
    """
    # 1. MIDI Handoff to music21
    # pretty_midi objects cannot be parsed directly by music21. 
    # We write the object to a temporary file on disk so music21 can read it.
    temp_filename = 'temp_presentation_live.mid'
    input_prettymidiobject.write(temp_filename)
        
    # Read and parse the MIDI file into an internal music21 Stream structure
    score = music21.converter.parse(temp_filename)

    # 2. Key Signature Detection
    # Analyze the distribution of pitch classes to find the most likely musical key.
    # This prevents the sheet music from being cluttered with endless individual sharp/flat symbols.
    detected_key = score.analyze('key') 
    
    # Inject the discovered key signature at the very beginning (offset 0) of every track/part
    for part in score.parts:
        part.insert(0, detected_key)
    
    # 3. Rhythm Quantization (Grid Alignment)
    # Human performances have micro-timing imperfections. 
    # quarterLengthDivisors=(2,) forces all note start-times (offsets) and lengths (durations) 
    # to snap strictly to the nearest 8th note grid, instantly cleaning up messy rhythmic clutter.
    clean_score = score.quantize(quarterLengthDivisors=(2,), processOffsets=True, processDurations=True)
    
    # 4. Initialize Empty Grand Staff Structure
    # Create the master canvas for a standard two-staff layout (Piano Grand Staff)
    piano_score = music21.stream.Score()
    
    # Setup the Right-Hand staff with a Treble Clef
    right_hand = music21.stream.Part()
    right_hand.append(music21.clef.TrebleClef()) 

    # Setup the Left-Hand staff with a Bass Clef
    left_hand = music21.stream.Part()
    left_hand.append(music21.clef.BassClef())   

    # 5. Process and Filter Every Note / Chord
    # .flatten() collapses multi-track structures into a single timeline, and .notes filters out rests
    for element in clean_score.flatten().notes:
        
        # --- Clean Note Durations ---
        # Humans release piano keys unpredictably, creating awkward, unreadable note lengths.
        # If a note length isn't a standard, clean beat value, we force-round it.
        if element.duration.quarterLength not in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            # Mathematical rounding to the nearest half-beat (8th note step)
            element.duration.quarterLength = round(element.duration.quarterLength * 2) / 2
            
            # Guardrail: If rounding shrinks a very short note down to 0, 
            # save it by forcing it to a minimum duration of an 8th note (0.5 beats).
            if element.duration.quarterLength == 0: 
                element.duration.quarterLength = 0.5 

        # --- Hands-Splitting Logic ---
        # Determine the target pitch to evaluate whether the element belongs to the left or right hand.
        if element.isChord:
            # For a cluster of notes (chord), use the lowest note to decide the hand assignment
            pitch_to_check = element.pitches[0].midi 
        else:
            # For a single note, grab its standard MIDI number
            pitch_to_check = element.pitch.midi
            
        # Middle C is MIDI note 60. 
        # Notes at or above Middle C go to the Treble Clef; lower notes go to the Bass Clef.
        if pitch_to_check >= 60:
            # Place the element into the right-hand staff at its original timeline position
            right_hand.insert(element.offset, element) 
        else:
            # Place the element into the left-hand staff at its original timeline position
            left_hand.insert(element.offset, element)  

    # 6. Assembly and Rendering
    # Attach both the completed right-hand and left-hand staves into the master score layout
    piano_score.append(right_hand)
    piano_score.append(left_hand)

    # Use the custom helper library 'showscore' to visually render the final notation inside the notebook
    return show(piano_score)