# Python Script to Generate CHORD_NOTES and CHORD_NOTES_OCTAVES Dictionaries

# Define the chromatic scale using only naturals and sharps
CHROMATIC_SCALE = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

# Define the chord types and their corresponding intervals in semitones
# Each interval is measured from the root note
CHORD_INTERVALS = {
    'maj': [0, 4, 7],           # Major Triad: root, major 3rd, perfect 5th
    'min': [0, 3, 7],           # Minor Triad: root, minor 3rd, perfect 5th
    'maj7': [0, 4, 7, 11],      # Major Seventh: root, major 3rd, perfect 5th, major 7th
    'min7': [0, 3, 7, 10],      # Minor Seventh: root, minor 3rd, perfect 5th, minor 7th
    'dim': [0, 3, 6],           # Diminished Triad: root, minor 3rd, diminished 5th
    'dom7': [0, 4, 7, 10],      # Dominant Seventh: root, major 3rd, perfect 5th, minor 7th
    'min7b5': [0, 3, 6, 10]     # Half-Diminished Seventh: root, minor 3rd, diminished 5th, minor 7th
}

def generate_chord_notes(root, chord_type):
    """
    Generates the fundamental notes of a chord based on the root and chord type.

    Parameters:
    - root (str): The root note (e.g., 'C', 'C#', 'D', etc.)
    - chord_type (str): The type of chord (e.g., 'maj', 'min', etc.)

    Returns:
    - chord_notes (list): A list of notes constituting the chord
    """
    if chord_type not in CHORD_INTERVALS:
        raise ValueError(f"Unsupported chord type: {chord_type}")

    intervals = CHORD_INTERVALS[chord_type]
    root_index = CHROMATIC_SCALE.index(root)
    chord_notes = []

    for interval in intervals:
        note_index = (root_index + interval) % len(CHROMATIC_SCALE)
        chord_notes.append(CHROMATIC_SCALE[note_index])

    return chord_notes

def assign_octaves(chord_notes, root):
    """
    Assigns octaves to each note in the chord, centering around octave 4.

    Parameters:
    - chord_notes (list): List of chord notes without octaves
    - root (str): The root note of the chord

    Returns:
    - chord_notes_octave (list): List of chord notes with assigned octaves
    """
    root_index = CHROMATIC_SCALE.index(root)
    chord_notes_octave = []

    for note in chord_notes:
        note_index = CHROMATIC_SCALE.index(note)
        if note == root:
            octave = 4
        elif note_index > root_index:
            octave = 4
        else:
            octave = 5
        chord_notes_octave.append(f"{note}{octave}")

    return chord_notes_octave

def main():
    # Initialize the dictionaries
    CHORD_NOTES = {}
    CHORD_NOTES_OCTAVES = {}

    # Define all root notes using naturals and sharps
    ROOT_NOTES = CHROMATIC_SCALE.copy()

    # Define all chord types
    CHORD_TYPES = ['maj', 'min', 'maj7', 'min7', 'dim', 'dom7', 'min7b5']

    # Generate the dictionaries
    for root in ROOT_NOTES:
        for chord_type in CHORD_TYPES:
            chord_name = f"{root}{chord_type}"
            # Generate fundamental chord notes
            chord_notes = generate_chord_notes(root, chord_type)
            CHORD_NOTES[chord_name] = chord_notes
            # Assign octaves to chord notes
            chord_notes_octave = assign_octaves(chord_notes, root)
            CHORD_NOTES_OCTAVES[chord_name] = chord_notes_octave

    # Optional: Print the CHORD_NOTES dictionary in a readable format
    print("CHORD_NOTES = {")
    for chord, notes in CHORD_NOTES.items():
        print(f"    '{chord}': {notes},")
    print("}\n")

    # Optional: Print the CHORD_NOTES_OCTAVES dictionary in a readable format
    print("CHORD_NOTES_OCTAVES = {")
    for chord, notes in CHORD_NOTES_OCTAVES.items():
        print(f"    '{chord}': {notes},")
    print("}")

if __name__ == "__main__":
    main()
