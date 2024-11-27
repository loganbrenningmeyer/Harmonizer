import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import re
import random

from _io import TextIOWrapper

from utils.data.mappings import *

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

'''
Custom Dataset to load inputs/labels by song,
rather than all songs at once
'''


class SongDataset(Dataset):
    def __init__(self, inputs_by_song, labels_by_song):
        self.inputs_by_song = inputs_by_song
        self.labels_by_song = labels_by_song

    def __len__(self):
        return len(self.inputs_by_song)
    
    def __getitem__(self, idx):
        return self.inputs_by_song[idx], self.labels_by_song[idx]


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


def get_songs_notes(ref_chords: bool = True):
    '''
    Returns:
    - notes_by_song: Nested list in the form: list[song_idx][note_idx]
        * e.g. [['C4', 'D4', 'D3'],
                ['C3', 'E3', 'G4'],
                ...]
    '''
    # -- Read chord_melody_data.txt
    with open('data/chord_melody_data.txt', 'r') as file:
        text = file.read()

    # -- Parse data into songs list of dicts
    songs = parse_data(text, ref_chords=ref_chords)

    # -- Extract note digit encodings from songs list
    songs_notes = [song['notes'] for song in songs]

    # # -- Convert note encodings to strings (note + octave)
    # songs_note_strings = [[NOTE_ENC_TO_NOTE_STR_REF.get(note_enc[:2]) + note_enc[2] for note_enc in song] 
    #                       for song in songs_note_encodings]

    return songs_notes


def create_dataloaders(ref_chords: bool = True):
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

    Returns:
    - train_dataloader
    - test_dataloader
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
            note_idx = NOTE_ENC_TO_IDX_REF.get(note[:2])
            # -- Obtain note one-hot encoding
            if note_idx is not None:
                note_input = note_enc_array[note_idx]
            else:
                note_input = np.zeros(num_notes, dtype=int)

            # -- Map chord to class label
            chord_label = CHORD_ENC_TO_IDX_REF.get(chord)

            # -- Append notes to inputs & chords to labels
            song_inputs.append(note_input)
            song_labels.append(chord_label)

        # -- Convert song inputs/labels to tensors
        song_inputs = torch.tensor(np.array(song_inputs, dtype=np.float32))
        song_labels = torch.tensor(np.array(song_labels, dtype=np.int64))

        # -- Append song's inputs/labels to full list
        inputs_by_song.append(song_inputs)
        labels_by_song.append(song_labels)

    # -- Randomly shuffle inputs/labels
    inputs_and_labels = list(zip(inputs_by_song, labels_by_song))
    
    random.seed(42)
    random.shuffle(inputs_and_labels)

    inputs_by_song, labels_by_song = zip(*inputs_and_labels)

    # -- Create training/testing sets (80% train/20% test)
    num_train = int(0.8 * len(inputs_by_song))

    train_inputs, train_labels = inputs_by_song[:num_train], labels_by_song[:num_train]
    test_inputs, test_labels = inputs_by_song[num_train:], labels_by_song[num_train:]

    # -- Create training TensorDataset/DataLoader
    train_dataset = SongDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # -- Create testing TensorDataset/DataLoader
    test_dataset = SongDataset(test_inputs, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, test_dataloader
