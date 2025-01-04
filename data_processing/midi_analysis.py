import mido
import numpy as np
from collections import defaultdict

def analyze_midi(midi_path):
    """
    Analyze a MIDI file and extract information useful for fingering annotations.
    Returns a list of notes with their timing and velocity information.
    """
    mid = mido.MidiFile(midi_path)
    
    # Convert MIDI ticks to seconds
    def ticks_to_seconds(ticks, ticks_per_beat, tempo):
        return ticks * tempo * 1e-6 / ticks_per_beat

    # Track note events and timing
    notes = []
    current_time = 0
    current_tempo = 500000  # Default tempo (120 BPM)
    active_notes = defaultdict(list)  # Track active notes per channel
    note_id = 0

    for track in mid.tracks:
        current_time = 0
        
        for msg in track:
            current_time += msg.time
            
            if msg.type == 'set_tempo':
                current_tempo = msg.tempo
            
            elif msg.type == 'note_on' and msg.velocity > 0:
                # Convert MIDI note number to pitch name
                note_num = msg.note
                octave = (note_num // 12) - 1
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                pitch_name = f"{note_names[note_num % 12]}{octave}"
                
                seconds = ticks_to_seconds(current_time, mid.ticks_per_beat, current_tempo)
                active_notes[msg.channel].append({
                    'id': note_id,
                    'start_time': seconds,
                    'pitch': pitch_name,
                    'midi_note': note_num,
                    'velocity': msg.velocity,
                    'channel': msg.channel
                })
                note_id += 1
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Find and update the matching note_on event
                for note in active_notes[msg.channel]:
                    if note['midi_note'] == msg.note:
                        seconds = ticks_to_seconds(current_time, mid.ticks_per_beat, current_tempo)
                        note['end_time'] = seconds
                        notes.append(note)
                        active_notes[msg.channel].remove(note)
                        break

    # Sort notes by start time
    notes.sort(key=lambda x: x['start_time'])
    
    # Analyze chord patterns and time intervals
    for i in range(len(notes)):
        # Look for simultaneous notes (potential chords)
        if i < len(notes) - 1:
            time_to_next = notes[i+1]['start_time'] - notes[i]['start_time']
            notes[i]['time_to_next'] = time_to_next
            notes[i]['likely_chord'] = time_to_next < 0.05  # Notes within 50ms are likely part of a chord
            
            # Find note range to help with hand distribution
            notes[i]['distance_to_next'] = abs(notes[i+1]['midi_note'] - notes[i]['midi_note'])
    
    return notes

def print_note_info(notes, output_file=None):
    """
    Print formatted note information to help with fingering decisions.
    If output_file is provided, save the information to that file.
    """
    # Create the output string
    output = []
    output.append("\nNote Analysis for Fingering:")
    output.append("-" * 80)
    output.append(f"{'ID':4} {'Start':8} {'End':8} {'Pitch':6} {'Velocity':8} {'Chord?':7} {'Distance':8}")
    output.append("-" * 80)
    
    for note in notes:
        chord_mark = '*' if note.get('likely_chord', False) else ''
        dist = f"{note.get('distance_to_next', '-'):3}" if 'distance_to_next' in note else '-'
        output.append(f"{note['id']:<4} {note['start_time']:8.2f} {note['end_time']:8.2f} "
                     f"{note['pitch']:<6} {note['velocity']:<8} {chord_mark:<7} {dist:<8}")
    
    # Add statistical information
    durations = [note['end_time'] - note['start_time'] for note in notes]
    velocities = [note['velocity'] for note in notes]
    chord_notes = sum(1 for note in notes if note.get('likely_chord', False))
    
    output.append("\nStatistical Information:")
    output.append(f"Total notes: {len(notes)}")
    output.append(f"Average note duration: {np.mean(durations):.2f} seconds")
    output.append(f"Average velocity: {np.mean(velocities):.2f}")
    output.append(f"Approximate number of chord notes: {chord_notes}")
    
    # Join all lines with newlines
    output_text = '\n'.join(output)
    
    # Print to console
    print(output_text)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_text)

if __name__ == "__main__":
    # Example usage
    midi_file = "/Users/almondgod/Repositories/robopianist/midi_files_cut/Guren no Yumiya Cut 14s.mid"
    output_file = "/Users/almondgod/Repositories/robopianist/data_processing/" + midi_file.rsplit('/', 1)[1].rsplit('.', 1)[0] + "_analysis.txt"  # Creates filename_analysis.txt
    
    notes = analyze_midi(midi_file)
    print_note_info(notes, output_file)