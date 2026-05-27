import music21
from showscore import show
import pretty_midi

def midi_to_note(input_notes, time_signature='4/4'):
    """
    Instantly converts pretty_midi object into highly-detailed,
    traditional Western sheet music directly inside the notebook.
    
    SHORTHAND NOTATION FORMAT RULES:
    - Add a length to a note to alter its rhythmic value:
      'C4' (Quarter/1 Beat), 'D2' (Half/2 Beats), 'E1' (Whole/4 Beats), 'F8' (Eighth/0.5 Beat)
    - Stacking notes into vertical chords using a comma: 'C4,E4,G4'
    """
    
    # 1. Parse Input Data
    if isinstance(input_notes, pretty_midi.PrettyMIDI):  # <-- Checks for the object type
        # Write the pretty_midi data to a temporary file so music21 can process it
        temp_filename = 'temp_presentation_live.mid'
        input_notes.write(temp_filename)
        
        # Parse it into the music21 score structure
        score = music21.converter.parse(temp_filename)
    else:
        # Otherwise, parse text strings into rhythmically sound music21 objects
        score = music21.stream.Score()
        part = music21.stream.Part()
        part.append(music21.meter.TimeSignature(time_signature))
        
        for item in input_notes.split():
            # Handle Chords (comma-separated entries)
            if ',' in item:
                chord_pitches = []
                # Check length from the last note element
                length_map = {'4': 1.0, '2': 2.0, '1': 4.0, '8': 0.5}
                q_len = length_map.get(item[-1], 1.0)
                
                clean_item = item[:-1] if item[-1].isdigit() else item
                chord_obj = music21.chord.Chord(clean_item.split(','))
                chord_obj.duration.quarterLength = q_len
                part.append(chord_obj)
            else:
                # Handle Notes
                length_map = {'4': 1.0, '2': 2.0, '1': 4.0, '8': 0.5}
                # Check if a rhythm token is supplied at the end of the note string
                if item[-1].isdigit() and item[-1] in length_map:
                    n = music21.note.Note(item[:-1])
                    n.duration.quarterLength = length_map[item[-1]]
                else:
                    n = music21.note.Note(item)
                    n.duration.quarterLength = 1.0 # Default to quarter note
                part.append(n)
                
        score.append(part)
        
    # 2. Render live, high-fidelity Western sheet music right inside the Jupyter UI!
    return show(score)