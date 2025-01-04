import re
from typing import Dict, List, Tuple

def parse_pitch(pitch_str):
    """Parse a pitch string (e.g. 'C#4') into note and octave."""
    match = re.match(r'([A-G][#b]?)(\d+)', pitch_str)
    if not match:
        raise ValueError(f"Invalid pitch format: {pitch_str}")
    note, octave = match.groups()
    return note, int(octave)

def is_right_hand(note, octave):
    """Determine if note should be played by right hand (>=C4)."""
    if octave > 4:
        return True
    if octave < 4:
        return False
    # For octave 4, compare with C4
    note_values = {'C': 0, 'C#': 0, 'Db': 0,
                'D': 1, 'D#': 1, 'Eb': 1,
                'E': 2, 'E#': 2, 'Fb': 2,
                'F': 3, 'F#': 3, 'Gb': 3,
                'G': 4, 'G#': 4, 'Ab': 4,
                'A': 5, 'A#': 5, 'Bb': 5,
                'B': 6, 'B#': 6, 'Cb': 6}
    return note_values.get(note, 0) >= 0  # C4 and above are right hand

def assign_finger(note):
        """
        Assign fingers based on RoboPianist convention:
        Right hand (>=C4): 0-4 (thumb to pinky)
        Left hand (<C4): 5-9 (thumb to pinky)
        """
        
        note, octave = parse_pitch(note['pitch'])
        
        if is_right_hand(note, octave):
            # Right hand fingering (0-4)
            if octave >= 5:
                return '3'  # Higher notes typically use middle finger (3)
            if note in ['A', 'A#', 'Bb', 'B', 'B#', 'Cb']:
                return '4'  # Pinky for higher notes in octave
            if note in ['G', 'G#', 'Ab']:
                return '3'  # Ring finger
            if note in ['E', 'F', 'F#', 'Gb']:
                return '2'  # Middle finger
            if note in ['C', 'C#', 'Db', 'D', 'D#', 'Eb']:
                return '1'  # Index/thumb for lower notes
            return '2'  # Default to middle finger
        else:
            # Left hand fingering (5-9)
            if octave <= 2:
                return '7'  # Lower notes typically use middle finger (7)
            if note in ['C', 'C#', 'Db', 'D', 'D#', 'Eb']:
                return '9'  # Pinky for lower notes in octave
            if note in ['E', 'F', 'F#', 'Gb']:
                return '8'  # Ring finger
            if note in ['G', 'G#', 'Ab', 'A', 'A#', 'Bb']:
                return '7'  # Middle finger
            if note in ['B', 'B#', 'Cb']:
                return '6'  # Index/thumb for higher notes
            return '7'  # Default to middle finger

def determine_hand(note, octave):
    """Return hand value for RoboPianist (0 for right, 1 for left)."""
    return 0 if is_right_hand(note, octave) else 1

def create_fingering_annotation(notes_data, output_file=None):
    """
    Convert note analysis data into RoboPianist fingering annotation format.
    RoboPianist uses:
    - Right hand: 0-4 (thumb to pinky)
    - Left hand: 5-9 (thumb to pinky)
    
    Args:
        notes_data: List of note dictionaries containing timing and pitch info
        output_file: Optional path to save the annotation. If None, returns as string
    """
    # Header
    output = "//Version: PianoFingering_v170101\n"
    
    # Process each note
    for i, note in enumerate(notes_data):
        start = float(note['start_time'])
        end = float(note['end_time'])
        pitch = note['pitch']
        velocity = note.get('velocity', 80)  # Default velocity 80 if not specified
        finger = assign_finger(note)
        
        # Format: id start end pitch velocity velocity hand finger
        note_name, octave = parse_pitch(pitch)
        hand = determine_hand(note_name, octave)
        line = f"{i}\t{start:.6f}\t{end:.6f}\t{pitch}\t{velocity}\t{velocity}\t{hand}\t{finger}\n"
        output += line
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
    
    return output

def parse_analysis_file(analysis_file: str) -> List[Dict]:
    """Parse the analysis file into the required format for fingering annotation."""
    notes = []
    parsing_notes = False
    
    with open(analysis_file, 'r') as f:
        for line in f:
            if line.startswith('ID'):
                parsing_notes = True
                continue
            if line.startswith('Statistical'):
                break
            if parsing_notes and line.strip() and not line.startswith('--'):
                # Split the line and extract data
                parts = line.strip().split()
                if len(parts) >= 7:  # Ensure we have all required fields
                    note = {
                        'id': int(parts[0]),
                        'start_time': float(parts[1]),
                        'end_time': float(parts[2]),
                        'pitch': parts[3],
                        'velocity': int(parts[4]),
                        'is_chord': parts[6] == '*' if len(parts) > 6 else False
                    }
                    notes.append(note)
    
    # Sort by start time and ID to maintain proper order
    notes.sort(key=lambda x: (x['start_time'], x['id']))
    return notes

# Example usage:
if __name__ == "__main__":
    # Parse the analysis file
    analysis_file = "/Users/almondgod/Repositories/robopianist/data_processing/Guren no Yumiya Cut 14s_analysis.txt"
    notes_data = parse_analysis_file(analysis_file)
    
    # Generate fingering annotation
    output_file = "Guren no Yumiya Cut 14s_fingering.txt"
    fingering = create_fingering_annotation(notes_data, output_file)
    
    print(f"Processed {len(notes_data)} notes")
    print(f"Fingering annotation saved to {output_file}")
    print("\nPreview of annotation:")
    print(fingering[:500] + "..." if len(fingering) > 500 else fingering)