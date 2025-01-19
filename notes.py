# Python Script to Generate HARMONIZING_NOTES Dictionary

# Define the chromatic scale using only naturals and sharps
CHROMATIC_SCALE = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

# Define the chord types and their corresponding scale intervals
# Intervals are in whole steps (2) and half steps (1)
scale_intervals = {
    'maj': [2, 2, 1, 2, 2, 2, 1],       # Major Scale
    'min': [2, 1, 2, 2, 1, 2, 2],       # Natural Minor Scale
    'maj7': [2, 2, 1, 2, 2, 2, 1],      # Major Scale
    'min7': [2, 1, 2, 2, 1, 2, 2],      # Natural Minor Scale
    'dim': [2, 1, 2, 1, 2, 1, 2],       # Whole-Half Diminished
    'dom7': [2, 2, 1, 2, 2, 1, 2],      # Mixolydian Mode
    'min7b5': [1, 2, 2, 1, 2, 2, 2]     # Locrian Mode
}

def generate_scale(root, intervals):
    """
    Generates a scale based on the root note and interval pattern.
    
    Parameters:
    - root (str): The root note (e.g., 'C', 'C#', 'D', etc.)
    - intervals (list): A list of integers representing whole (2) and half (1) steps
    
    Returns:
    - scale (list): A list of notes in the generated scale
    """
    scale = [root]
    start_index = CHROMATIC_SCALE.index(root)
    current_index = start_index
    for step in intervals[:-1]:  # The last step brings us back to the octave
        current_index = (current_index + step) % len(CHROMATIC_SCALE)
        scale.append(CHROMATIC_SCALE[current_index])
    return scale

# Initialize the HARMONIZING_NOTES dictionary
HARMONIZING_NOTES = {}

# Define all root notes using naturals and sharps
ROOT_NOTES = CHROMATIC_SCALE.copy()

# Define all chord types
CHORD_TYPES = ['maj', 'min', 'maj7', 'min7', 'dim', 'dom7', 'min7b5']

# Generate the harmonizing notes for each root and chord type
for root in ROOT_NOTES:
    for chord_type in CHORD_TYPES:
        chord_name = f"{root}{chord_type}"
        intervals = scale_intervals[chord_type]
        scale = generate_scale(root, intervals)
        HARMONIZING_NOTES[chord_name] = scale

# Optional: Print the HARMONIZING_NOTES dictionary in a readable format
for chord, notes in HARMONIZING_NOTES.items():
    print(f"'{chord}': {notes},")
