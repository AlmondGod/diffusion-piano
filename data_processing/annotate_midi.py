import re
from typing import Dict, List, Tuple
from collections import defaultdict

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

def assign_finger(note, chord_notes=None):
    """
    Assign fingers based on RoboPianist convention with chord awareness:
    Right hand (>=C4): 0-4 (thumb to pinky)
    Left hand (<C4): 5-9 (thumb to pinky)
    
    Args:
        note: Dictionary containing note information
        chord_notes: List of notes that are part of the same chord
    """
    note_name, octave = parse_pitch(note['pitch'])
    
    # Get all simultaneous notes if chord_notes is provided
    if chord_notes:
        chord_pitches = [n['pitch'] for n in chord_notes]
        chord_size = len(chord_pitches)
    else:
        chord_size = 1
    
    if is_right_hand(note_name, octave):
        # Right hand fingering (0-4)
        if chord_size >= 3 and note.get('is_chord', False):
            # For chords with 3 or more notes, use specific fingering patterns
            if note_name in ['C', 'C#', 'Db', 'D', 'D#', 'Eb']:
                return '1'  # Thumb for bottom notes
            if note_name in ['G', 'G#', 'Ab', 'A', 'A#', 'Bb']:
                return '3'  # Middle finger for middle notes
            if octave >= 5:
                return '5'  # Pinky for top notes
        else:
            # Regular fingering for non-chord or small chord notes
            if octave >= 5:
                return '3'
            if note_name in ['A', 'A#', 'Bb', 'B', 'B#', 'Cb']:
                return '4'
            if note_name in ['G', 'G#', 'Ab']:
                return '3'
            if note_name in ['E', 'F', 'F#', 'Gb']:
                return '2'
            if note_name in ['C', 'C#', 'Db', 'D', 'D#', 'Eb']:
                return '1'
        return '2'  # Default to middle finger
    else:
        # Left hand fingering (5-9)
        if chord_size >= 3 and note.get('is_chord', False):
            # For left hand chords
            if note_name in ['C', 'C#', 'Db', 'D', 'D#', 'Eb']:
                return '9'  # Pinky for bottom notes
            if note_name in ['G', 'G#', 'Ab', 'A', 'A#', 'Bb']:
                return '7'  # Middle finger for middle notes
            if octave >= 3:
                return '5'  # Thumb for top notes
        else:
            # Regular fingering for non-chord or small chord notes
            if octave <= 2:
                return '7'
            if note_name in ['C', 'C#', 'Db', 'D', 'D#', 'Eb']:
                return '9'
            if note_name in ['E', 'F', 'F#', 'Gb']:
                return '8'
            if note_name in ['G', 'G#', 'Ab', 'A', 'A#', 'Bb']:
                return '7'
            if note_name in ['B', 'B#', 'Cb']:
                return '6'
        return '7'  # Default to middle finger

def determine_hand(note, octave):
    """Return hand value for RoboPianist (0 for right, 1 for left)."""
    return 0 if is_right_hand(note, octave) else 1

def create_fingering_annotation(notes_data, output_file=None):
    """
    Convert note analysis data into RoboPianist fingering annotation format.
    Process all notes, including chord notes.
    """
    # Header
    output = "//Version: PianoFingering_v170101\n"
    
    # Group notes by start time to identify chords
    notes_by_time = defaultdict(list)
    for note in notes_data:
        start_time = float(note['start_time'])
        notes_by_time[start_time].append(note)
    
    # Sort notes by start time and pitch
    sorted_notes = sorted(notes_data, key=lambda x: (x['start_time'], x['pitch']))
    
    # Process each note
    for i, note in enumerate(sorted_notes):
        start = float(note['start_time'])
        end = float(note['end_time'])
        pitch = note['pitch']
        velocity = note.get('velocity', 80)
        
        # Get all notes in the same chord
        chord_notes = notes_by_time[start]
        finger = assign_finger(note, chord_notes)
        
        note_name, octave = parse_pitch(pitch)
        hand = determine_hand(note_name, octave)
        line = f"{i}\t{start:.6f}\t{end:.6f}\t{pitch}\t{velocity}\t{velocity}\t{hand}\t{finger}\n"
        output += line
    
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
                # Remove multiple spaces and split
                parts = ' '.join(line.strip().split()).split()
                
                # Skip if we don't have at least the basic fields
                if len(parts) < 5:
                    continue
                    
                try:
                    note = {
                        'id': int(parts[0]),
                        'start_time': float(parts[1]),
                        'end_time': float(parts[2]),
                        'pitch': parts[3],
                        'velocity': int(parts[4]),
                        'is_chord': '*' in line,
                        'distance_to_next': int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else None
                    }
                    notes.append(note)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
                    continue
    
    # Sort by start time and ID
    notes.sort(key=lambda x: (x['start_time'], x['id']))
    
    # Debug output
    print(f"Found {len(notes)} notes in analysis file")
    print(f"First few notes: {notes[:3]}")
    
    return notes

# Example usage:
if __name__ == "__main__":
    # Parse the analysis file
    analysis_file = "/Users/almondgod/Repositories/robopianist/data_processing/Guren no Yumiya Cut 14s_analysis.txt"
    notes_data = parse_analysis_file(analysis_file)
    print(f"Parsed {len(notes_data)} notes from analysis file")
    
    # Generate fingering annotation
    output_file = "Guren no Yumiya Cut 14s_fingering v3.txt"
    fingering = create_fingering_annotation(notes_data, output_file)
    
    # Verify output
    with open(output_file, 'r') as f:
        output_lines = [line for line in f if not line.startswith('//')]
    print(f"Generated {len(output_lines)} note annotations")