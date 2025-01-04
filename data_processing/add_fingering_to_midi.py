from note_seq import NoteSequence, midi_io
from typing import List, Tuple, Union
from pathlib import Path
import re
from robopianist.music.midi_file import MidiFile

def parse_pitch_to_midi_number(pitch_str: str) -> int:
    """Convert pitch string (e.g., 'C#4') to MIDI note number."""
    note_values = {
        'C': 0, 'C#': 1, 'Db': 1,
        'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11
    }
    
    match = re.match(r'([A-G][#b]?)(\d+)', pitch_str)
    if not match:
        raise ValueError(f"Invalid pitch format: {pitch_str}")
    
    note, octave = match.groups()
    return note_values[note] + (int(octave) + 1) * 12

def add_fingering_from_annotation_file(
    input_midi: Union[str, Path],
    annotation_file: Union[str, Path]
) -> None:
    """
    Add fingering annotations from the RoboPianist fingering format file. Return modified note sequence object.
    Format: id start_time end_time pitch velocity velocity hand finger
    
    Fingering convention:
    - Right hand: 0-4 (thumb to pinky)
    - Left hand: 5-9 (thumb to pinky)
    """
    # Make paths absolute
    input_midi = Path(input_midi).absolute()
    annotation_file = Path(annotation_file).absolute()
    
    print(f"Loading MIDI from: {input_midi}")
    print(f"Loading annotations from: {annotation_file}")
    
    # Read annotation file
    fingering_data = []
    with open(annotation_file, 'r') as f:
        for line in f:
            if line.startswith('//') or not line.strip():
                continue
            
            parts = line.strip().split('\t')
            if len(parts) == 8:
                _, start, end, pitch, _, _, hand, finger = parts
                midi_number = parse_pitch_to_midi_number(pitch)
                finger_value = int(finger)
                if 0 <= finger_value <= 9:  # Validate finger value
                    fingering_data.append((
                        float(start),
                        float(end),
                        midi_number,
                        finger_value 
                    ))
                    print(f"Added fingering {finger_value} for note {pitch} at {start}")
    
    sequence = midi_io.midi_file_to_note_sequence(str(input_midi))
    
    # Add fingering information to each note
    for note in sequence.notes:
        for start, end, pitch, finger in fingering_data:
            if (abs(note.start_time - start) < 0.01 and 
                abs(note.end_time - end) < 0.01 and 
                note.pitch == pitch):
                note.part = finger 
                break
    
    # print total number of notes and number of notes with fingering
    print(f"Total number of notes: {len(sequence.notes)}")
    print(f"Number of notes with fingering: {sum(1 for note in sequence.notes if note.part is not 0)}")

    # conver to midi_file object
    midi = MidiFile(seq=sequence)
    return midi

if __name__ == "__main__":
    midi = add_fingering_from_annotation_file(
        "/Users/almondgod/Repositories/robopianist/midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "/Users/almondgod/Repositories/robopianist/data_processing/Guren no Yumiya Cut 14s_fingering v3.txt"
    )

    print(f"Has fingering: {midi.has_fingering()}")
