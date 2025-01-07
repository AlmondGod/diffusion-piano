import mido
from mido import MidiFile
import argparse

def note_number_to_name(note_number):
    """Convert MIDI note number to note name."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    note = notes[note_number % 12]
    # Replace sharp notes with flat equivalents
    if '#' in note:
        flat_notes = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
        note = flat_notes[note]
    return f"{note}{octave}"

def convert_midi_to_text(midi_file_path, output_file_path):
    """Convert MIDI file to text format with timing and placeholder annotations."""
    mid = MidiFile(midi_file_path)
    
    # Track note events
    notes = []
    current_time = 0
    note_status = {}  # Keep track of active notes
    
    # Process all MIDI tracks
    for track in mid.tracks:
        track_time = 0
        for msg in track:
            track_time += msg.time
            current_time = track_time * mid.ticks_per_beat * (60 / 120) / 480  # Convert ticks to seconds assuming 120 BPM
            
            if msg.type == 'note_on' and msg.velocity > 0:
                # Note start
                note_status[msg.note] = {
                    'start_time': current_time,
                    'velocity': msg.velocity,
                    'channel': msg.channel
                }
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                # Note end
                if msg.note in note_status:
                    start_info = note_status[msg.note]
                    notes.append({
                        'start_time': start_info['start_time'],
                        'end_time': current_time,
                        'note': msg.note,
                        'velocity': start_info['velocity'],
                        'channel': start_info['channel']
                    })
                    del note_status[msg.note]
    
    # Sort notes by start time
    notes.sort(key=lambda x: x['start_time'])
    
    # Write to output file
    with open(output_file_path, 'w') as f:
        for i, note in enumerate(notes):
            note_name = note_number_to_name(note['note'])
            # Default values for hand (0/1) and finger (placeholder numbers)
            hand = 0 if note['channel'] == 0 else 1
            finger = (i % 5) + 1 if hand == 0 else -(i % 5) - 1  # Simple placeholder finger numbers
            
            # Format: index start_time end_time note velocity hand finger
            f.write(f"{i}\t{note['start_time']:.6f}\t{note['end_time']:.6f}\t"
                   f"{note_name}\t{64}\t{note['velocity']}\t{hand}\t{finger}\n")

def main():
    # parser = argparse.ArgumentParser(description='Convert MIDI file to text format with timing and annotations')
    # parser.add_argument('input_file', help='Input MIDI file path', default='midi_files/Crossing Field.mid')
    # parser.add_argument('output_file', help='Output text file path', default='./Crossing_Field_Fingering.txt')
    
    # args = parser.parse_args()
    
    # try:
    #     convert_midi_to_text(args.input_file, args.output_file)
    #     print(f"Successfully converted {args.input_file} to {args.output_file}")
    # except Exception as e:
    #     print(f"Error: {str(e)}")
    convert_midi_to_text('/Users/almondgod/Repositories/robopianist/midi_files_cut/Crossing Field Cut 10s.mid', './Crossing_Field_Fingering.txt')

if __name__ == "__main__":
    main()