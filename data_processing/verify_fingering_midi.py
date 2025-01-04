from robopianist.music.midi_file import NoteTrajectory

# Create note trajectory to see the actual notes with fingering
note_traj = NoteTrajectory.from_midi("/Users/almondgod/Repositories/robopianist/Guren No Yumiya Cut 14s Annotated.mid", dt = 0.01)
notes = note_traj.notes

# Print first few timesteps of notes with their fingering
for t in range(min(5, len(notes))):
    print(f"\nTimestep {t}:")
    for note in notes[t]:
        print(f"Key: {note.key}, Fingering: {note.fingering}")