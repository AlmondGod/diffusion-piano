from mido import MidiFile, MidiTrack, Message, MetaMessage
import copy

def bpm2tempo(bpm):
    """Convert beats per minute (BPM) to microseconds per beat."""
    return int(60 * 1000000 / bpm)

def get_tempo(midi_file):
    """Get the tempo from the MIDI file. Returns default 120 BPM if not found."""
    for track in midi_file.tracks:
        for msg in track:
            if isinstance(msg, MetaMessage) and msg.type == 'tempo':
                return msg.tempo
    return bpm2tempo(120)  # Default 120 BPM

def ticks_to_seconds(ticks, tempo, ticks_per_beat):
    """Convert ticks to seconds based on tempo and ticks per beat."""
    return (ticks * tempo) / (ticks_per_beat * 1000000)

def clip_midi(input_file, output_file, max_seconds):
    """
    Clip a MIDI file to a specified length in seconds.
    
    Args:
        input_file (str): Path to input MIDI file
        output_file (str): Path to save the clipped MIDI file
        max_seconds (float): Maximum length in seconds
    """
    # Read the input MIDI file
    midi_in = MidiFile(input_file)
    
    # Create a new MIDI file with the same ticks_per_beat
    midi_out = MidiFile(ticks_per_beat=midi_in.ticks_per_beat)
    
    # Get the tempo (microseconds per beat)
    tempo = get_tempo(midi_in)
    
    # Process each track
    for track_in in midi_in.tracks:
        track_out = MidiTrack()
        midi_out.tracks.append(track_out)
        
        absolute_time = 0.0  # Track absolute time in seconds
        accumulated_ticks = 0  # Track accumulated ticks
        
        # Copy time signature and tempo messages from the start
        for msg in track_in:
            if isinstance(msg, MetaMessage) and msg.type in ['time_signature', 'tempo', 'key_signature']:
                track_out.append(msg)
                if msg.type == 'tempo':
                    tempo = msg.tempo
        
        # Copy messages until we reach max_seconds
        for msg in track_in:
            # Calculate absolute time more precisely
            if hasattr(msg, 'time'):
                accumulated_ticks += msg.time
                absolute_time = ticks_to_seconds(accumulated_ticks, tempo, midi_in.ticks_per_beat)
            
            # Update tempo if we encounter a tempo change
            if isinstance(msg, MetaMessage) and msg.type == 'tempo':
                tempo = msg.tempo
            
            # If we're past the max time, stop adding messages
            if absolute_time > max_seconds:
                # Add note-off messages for any currently playing notes
                if hasattr(msg, 'type') and msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                    note_off = Message('note_off', note=msg.note, velocity=0, time=0, channel=msg.channel)
                    track_out.append(note_off)
                break
            
            # Copy the message
            track_out.append(msg)
        
        # Add end of track message
        track_out.append(MetaMessage('end_of_track'))
    
    # Save the output file
    midi_out.save(output_file)

# Example usage:
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_midi = "/Users/almondgod/Repositories/robopianist/midi_files/Sword Art Online - Crossing Field (SAO OP).mid"
    output_midi = "/Users/almondgod/Repositories/robopianist/midi_files_cut/Crossing Field Cut 10s.mid"
    clip_duration = 15  # Clip to 10 seconds
    
    clip_midi(input_midi, output_midi, clip_duration)