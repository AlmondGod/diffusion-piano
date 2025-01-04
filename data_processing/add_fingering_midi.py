from note_seq import NoteSequence, midi_io
from typing import List, Tuple, Union
from pathlib import Path

def add_fingering_to_midi(
    input_midi: Union[str, Path],
    output_midi: Union[str, Path],
    fingering_data: List[Tuple[float, float, str, int]]
) -> None:
    """
    Add fingering annotations to a MIDI file.
    
    Args:
        input_midi: Path to input MIDI file
        output_midi: Path to save annotated MIDI file
        fingering_data: List of tuples containing (start_time, end_time, pitch, fingering)
            where fingering is:
            - 0-4 for right hand (thumb to pinky)
            - 5-9 for left hand (thumb to pinky)
    """
    # Load the MIDI file
    sequence = midi_io.midi_file_to_note_sequence(str(input_midi))
    
    # Create a mapping of (start_time, end_time, pitch) to fingering
    fingering_map = {(s, e, p): f for s, e, p, f in fingering_data}
    
    # Add fingering information to each note
    for note in sequence.notes:
        key = (note.start_time, note.end_time, note.pitch)
        if key in fingering_map:
            note.part = fingering_map[key]
    
    # Save the modified MIDI file
    midi_io.note_sequence_to_midi_file(sequence, str(output_midi))

def add_fingering_from_annotation_file(
    input_midi: Union[str, Path],
    output_midi: Union[str, Path],
    annotation_file: Union[str, Path]
) -> None:
    """
    Add fingering annotations from a text file in the format:
    start_time end_time pitch fingering
    
    Args:
        input_midi: Path to input MIDI file
        output_midi: Path to save annotated MIDI file
        annotation_file: Path to annotation file
    """
    # Read annotation file
    fingering_data = []
    with open(annotation_file, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith('//') or not line.strip():
                continue
            
            # Parse the line
            try:
                start, end, pitch, _, _, _, fingering = line.strip().split('\t')
                fingering_data.append((
                    float(start),
                    float(end),
                    pitch,  # Note name (e.g., 'C4')
                    int(fingering) if fingering != '-' else -1
                ))
            except ValueError:
                print(f"Skipping malformed line: {line.strip()}")
    
    add_fingering_to_midi(input_midi, output_midi, fingering_data)

# Example usage:
if __name__ == "__main__":
    # # Method 1: Direct fingering data
    # fingering_example = [
    #     (0.0, 0.5, "C4", 1),  # Right hand index finger
    #     (0.5, 1.0, "E4", 3),  # Right hand middle finger
    #     (1.0, 1.5, "G4", 5),  # Left hand thumb
    # ]
    
    # # Add fingering directly
    # add_fingering_to_midi(
    #     "input.mid",
    #     "output_annotated.mid",
    #     fingering_example
    # )
    
    # Method 2: From annotation file
    add_fingering_from_annotation_file(
        "/Users/almondgod/Repositories/robopianist/midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "Guren No Yumiya Cut 14s Annotated.mid",
        "/Users/almondgod/Repositories/robopianist/data_processing/Guren no Yumiya Cut 14s_fingering.txt"
    )