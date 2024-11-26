import torch
import numpy as np
import os
import re

from _io import TextIOWrapper

from mappings import NOTES_REF, CHORDS_REF

'''
chord_melody_data.txt format:

[Key]^
[Chord1]*[Chord2]*...[Chord8]*
[Notes for Bar1]+[Notes for Bar2]+[Notes for Bar3]+[Notes for Bar4]#

- Group of 4 bars at a time, each group separated by a #
- Each group of 4 bars has a two-digit key
    * Key: 62^
- Each bar within the group has two chords:
    * Bar1: [Chord1]*[Chord2]*
    * Bar2: [Chord3]*[Chord4]*
    * Bar3: [Chord5]*[Chord6]*
    * Bar4: [Chord7]*[Chord8]*
- The notes are separated by + for each bar (last bar ends with #):
    * Bar1: 6250-6251-3150-3151-3151-3151-3151-3151-1140-1141-2140-2141-1140-1141-6240-6241+
    * Bar2: 1140-1141-3250-3251-0000-0000-0000-0000-0000-0000-0000-0000-0000-0000-0000-0000+
    * Bar3: 6250-6251-3250-3251-3251-3251-3251-3251-1140-1141-2140-2141-1140-1141-6240-6241+
    * Bar4: 1140-1141-6240-6241-0000-0000-0000-0000-0000-0000-0000-0000-0000-0000-0000-0000#

Break up songs at each key change:
- i.e. group consecutive measures with the same key into the same song

Batch songs:
- Can either batch songs together in order without shuffling, or do each song individually (batch = 1)
'''
def is_compatible(song: dict):
    '''
    Verifies that a song only uses major or dom7 chords of natural notes
    to ensure compatibility with reference model architecture
    '''
    for chord in song['chords']:
        # If not a natural root note or not maj/dom7
        if chord[1] != '1' or (chord[2] != '0' and chord[2] != '5'):
            return False
    
    return True


def parse_data(text: str, ref_chords: bool = True):
    '''
    Parses the chord_melody_data.txt file for songs, 
    storing the notes/chords at the 1st/3rd beats

    Separates songs in the data .txt file at key changes,
    assuming each song has a different key

    Parameters:
    - text: String of the full chord_melody_data.txt 
      file returned from file.read()
    - ref_chords: Boolean defining if only chords
      compatible with the reference architecture
      (maj & dom7) should be used

    Returns:
    - songs: List of dictionaries for each song containing
      the 'notes' and 'chords' at the 1st/3rd beats
    '''
    # -- Split text file into 4-bar groups
    bar_groups = text.strip().split('#')[:-1]

    # -- Define regex to extract key, chords, and notes for each 4-bar group
    pattern = re.compile(r'^(?P<key>\d{2})\^(?P<chords>(\d{3}\*){8})(?P<notes>.+)$')

    # -- Initialize array to store data for each song
    songs = []

    # -- Initialize key/song dict to determine song changes
    curr_key = None
    # song = dict of 1st/3rd beat notes/chords for full song
    song = {
        'notes': [],
        'chords': []
    }

    for bar_group in bar_groups:
        # -- Use regex to extract key, chords, and notes
        re_match = pattern.match(bar_group)

        # -- Split key, chords, and notes from regex match
        key = re_match.group('key')
        chords = re_match.group('chords').split('*')[:-1]
        notes = [bar_notes.split('-') for bar_notes in re_match.group('notes').split('+')]

        # -- Begin a new song when the key changes
        if key != curr_key:
            # If it isn't the first iteration...
            if curr_key is not None:
                # If only using reference model chords (maj & dom7) check compatibility
                if ref_chords:
                    if is_compatible(song):
                        songs.append(song)
                # If not using reference model chords, return any song
                else:
                    songs.append(song)
            
            # Reset song dict
            song = {
                'notes': [],
                'chords': []
            }

            # Update curr_key
            curr_key = key

        # -- Append 1st/3rd beat chords to song dict (all chords)
        song['chords'].extend(chords)

        # -- Append 1st/3rd beat notes to song dict
        for bar_notes in notes:
            song['notes'].extend([bar_notes[0], bar_notes[8]])

    return songs


def create_training_data(ref_chords: bool = True):
    '''
    Conversed parsed song data to one-hot encodings
    for notes/chords
    - notes: One-hot encoding of chromatic scale

        * [A  A#/Bb B  C  C#/Db D  D#/Eb E  F  F#/Gb G   G#/Ab]
        * [0  1     2  3  4     5  6     7  8  9     10  11   ]
    - chords (ref_chords = True): One-hot encoding of 14 chords (7 maj, 7 dom7)
        * [Amaj  Bmaj  Cmaj  Dmaj  Emaj  Fmaj  Gmaj  Adom7 Bdom7 Cdom7 Ddom7 Edom7 Fdom7 Gdom7]
        * [110   210   310   410   510   610   710   115   215   315   415   515   615   715  ]
    - chords (ref_chords = False): One-hot encoding of 84 chords (12 chromatic notes * 7 chord types)
        * [A->G#] * maj, min, dim, maj7, min7, dom7, min7b5
    '''
    # -- Read chord_melody_data
    with open('data/chord_melody_data.txt', 'r') as file:
        text = file.read()

    # -- Parse data into songs list of dicts
    songs = parse_data(text, ref_chords=ref_chords)

    # -- Define notes/chords one-hot size 
    num_notes = 12

    if ref_chords:
        num_chords = 14
    else:
        num_chords = 84

    # -- Initialize note/chord one-hot encoding arrays to index from
    note_enc_array = np.eye(num_notes, dtype=int)

    # -- Initialize empty inputs/labels arrays
    inputs_by_song = []
    labels_by_song = []
    
    for song in songs:
        song_inputs = []
        song_labels = []

        for note, chord in zip(song['notes'], song['chords']):
            # -- Map note to one-hot index
            note_idx = NOTES_REF.get(note[:2])
            # -- Obtain note one-hot encoding
            if note_idx is not None:
                note_input = note_enc_array[note_idx]
            else:
                note_input = np.zeros(num_notes, dtype=int)

            # -- Map chord to class label
            chord_label = CHORDS_REF.get(chord)

            # -- Append notes to inputs & chords to labels
            song_inputs.append(note_input)
            song_labels.append(chord_label)

        # -- Convert song inputs/labels to tensors
        song_inputs = torch.tensor(np.array(song_inputs, dtype=np.float32))
        song_labels = torch.tensor(np.array(song_labels, dtype=np.int64))

        # -- Append song's inputs/labels to full list
        inputs_by_song.append(song_inputs)
        labels_by_song.append(song_labels)

    return inputs_by_song, labels_by_song
