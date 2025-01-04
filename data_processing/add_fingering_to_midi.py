from note_seq import NoteSequence, midi_io
from typing import List, Tuple, Union
from pathlib import Path
import re

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
    output_midi: Union[str, Path],
    annotation_file: Union[str, Path]
) -> None:
    """
    Add fingering annotations from the RoboPianist fingering format file.
    Format: id start_time end_time pitch velocity velocity hand finger
    
    Args:
        input_midi: Path to input MIDI file
        output_midi: Path to save annotated MIDI file
        annotation_file: Path to annotation file
    """
    # Read annotation file
    fingering_data = []
    with open(annotation_file, 'r') as f:
        for line in f:
            # Skip comments, empty lines, and header
            if line.startswith('//') or not line.strip():
                continue
            
            # Parse the line
            try:
                parts = line.strip().split('\t')
                if len(parts) == 8:  # Make sure we have all fields
                    _, start, end, pitch, vel1, vel2, hand, finger = parts
                    midi_number = parse_pitch_to_midi_number(pitch)
                    fingering_data.append((
                        float(start),
                        float(end),
                        midi_number,
                        int(finger)
                    ))
            except (ValueError, IndexError) as e:
                print(f"Skipping malformed line: {line.strip()}, Error: {e}")
    
    # Load the MIDI file
    sequence = midi_io.midi_file_to_note_sequence(str(input_midi))
    
    # Create a mapping of (start_time, end_time, midi_number) to fingering
    fingering_map = {(s, e, p): f for s, e, p, f in fingering_data}
    
    # Add fingering information to each note
    for note in sequence.notes:
        key = (note.start_time, note.end_time, note.pitch)
        # Try to find an approximate match if exact timing doesn't match
        if key not in fingering_map:
            # Look for notes with similar timing (within 0.01 seconds)
            for (s, e, p), f in fingering_map.items():
                if (abs(s - note.start_time) < 0.01 and 
                    abs(e - note.end_time) < 0.01 and 
                    p == note.pitch):
                    note.part = f
                    break
        else:
            note.part = fingering_map[key]
    
    # Save the modified MIDI file
    midi_io.note_sequence_to_midi_file(sequence, str(output_midi))

if __name__ == "__main__":
    add_fingering_from_annotation_file(
        "/Users/almondgod/Repositories/robopianist/midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "Guren No Yumiya Cut 14s Annotated.mid",
        "/Users/almondgod/Repositories/robopianist/data_processing/Guren no Yumiya Cut 14s_fingering.txt"
    )