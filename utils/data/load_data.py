import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import random
import seaborn as sns
import pandas as pd

from _io import TextIOWrapper

from utils.data.mappings import *
from utils.data.distributions import balance_samples

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


def parse_data(hnn_data: bool):
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
    # -- Read chord_melody_data
    with open('data/chord_melody_data.txt', 'r') as file:
        text = file.read()

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
        'key': None,
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

            song['key'] = key

            # If it isn't the first iteration...
            if curr_key is not None:
                # If only using reference model chords (maj & dom7) check compatibility
                if hnn_data:
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

        # HNN Model (Melody -> Chord)
        if hnn_data:
            # -- Append 1st/3rd beat chords to song dict (all chords)
            song['chords'].extend(chords)

            # -- Append 1st/3rd beat notes to song dict
            for bar_notes in notes:
                song['notes'].extend([bar_notes[0], bar_notes[8]])

        # MelodyNet Model (Chord -> Melody)
        else:
            '''
            Want pairs of chords and notes w/ duration instead of lifespan

            e.g. ('426', '6251'), ('426', '315)
            '''
            # -- Append all chords duplicated 8 times to cover all 16th beat timesteps (2 chords per measure)
            song['chords'].extend([chord for chord in chords for _ in range(8)])

            # -- Append all notes 
            song['notes'].extend([note for bar_notes in notes for note in bar_notes])

    return songs


def get_songs_chords():
    # -- Parse data into songs list of dicts
    songs = parse_data(hnn_data=False)

    # -- Extract chord digit encodings from songs list
    songs_chords = [song['chords'] for song in songs]

    return songs_chords


def get_songs_notes():
    '''
    Returns:
    - notes_by_song: Nested list in the form: list[song_idx][note_idx]
        * e.g. [['C4', 'D4', 'D3'],
                ['C3', 'E3', 'G4'],
                ...]
    '''
    # -- Parse data into songs list of dicts
    songs = parse_data(hnn_data=True)

    # -- Extract note digit encodings from songs list
    songs_notes = [song['notes'] for song in songs]

    return songs_notes


def create_dataloaders_hnn():
    '''
    Creates training/testing dataloaders for the HNN model

    - notes: One-hot encoding of chromatic scale
        * [A  A# B  C  C# D  D# E  F  F# G  G#]
        * [0  1  2  3  4  5  6  7  8  9  10 11]
    - chords: One-hot encoding of 14 chords (7 maj, 7 dom7)
        * [Amaj  Bmaj  Cmaj  Dmaj  Emaj  Fmaj  Gmaj  Adom7 Bdom7 Cdom7 Ddom7 Edom7 Fdom7 Gdom7]
        * [0     1     2     3     4     5     6     7     8     9     10    11    12    13  ]

    Returns:
    - train_dataloader
    - test_dataloader
    '''
    # -- Parse data into songs list of dicts
    songs = parse_data(hnn_data=True)

    # -- Define notes/chords one-hot size 
    num_notes = 12

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


# def create_dataloaders_mnet():
#     '''
#     Creates training/testing dataloaders for MelodyNet model

#     - chords: One-hot encoding of 84 possible chords (12 notes * 7 types)
#         * [Amaj/min/dim/maj7/min7/dom7/min7b5 -> G#maj/min/dim/maj7/min7/dom7/min7b5]
#         * [0    1   2   3    4    5    6     ... 77    78  79  80   81   82   83    ]
#     - notes: Labels of 145 possible note/octave/lifespan combinations
#         * [rest A2(0) A3(0) ... A6(1) A7(1) ... G#6(1) G#7(1)]
#         * [0    1     2     ... 11    12    ... 143    144   ]
#     '''
#     # -- Parse data into songs list of dicts
#     songs = parse_data(hnn_data=False)

#     # -- Define chords one-hot size 
#     num_chords = 84

#     # -- Initialize note/chord one-hot encoding arrays to index from
#     chord_enc_array = np.eye(num_chords, dtype=int)

#     # -- Initialize empty inputs/labels arrays
#     inputs_by_song = []
#     labels_by_song = []

#     for song in songs:
#         song_inputs = []
#         song_labels = []

#         for note, chord in zip(song['notes'], song['chords']):
#             # -- Map chord to one-hot index (7 * note_idx + chord_type_idx)
#             chord_idx = 7 * NOTE_ENC_TO_IDX_REF.get(chord[:2]) + int(chord[2])
#             # -- Obtain chord input one-hot encoding
#             chord_input = chord_enc_array[chord_idx]

#             # -- Map note to class label
#             note_idx = NOTE_ENC_TO_IDX_REF.get(note[:2])
#             note_octave = int(note[2])
#             note_lifespan = int(note[3])
#             # Rest note
#             if note_idx is None:
#                 note_label = 0
#             else:
#                 note_label = 1 + (note_idx * 12) + (note_octave - 2) + (note_lifespan * 6)

#             # -- Append chord to inputs & note to labels
#             song_inputs.append(chord_input)
#             song_labels.append(note_label)

#         # -- Convert song inputs/labels to tensors
#         song_inputs = torch.tensor(np.array(song_inputs, dtype=np.float32))
#         song_labels = torch.tensor(np.array(song_labels, dtype=np.int64))

#         # -- Append song's inputs/labels to full list
#         inputs_by_song.append(song_inputs)
#         labels_by_song.append(song_labels)

#     # -- Randomly shuffle inputs/labels
#     inputs_and_labels = list(zip(inputs_by_song, labels_by_song))
    
#     random.seed(42)
#     random.shuffle(inputs_and_labels)

#     inputs_by_song, labels_by_song = zip(*inputs_and_labels)

#     # -- Create training/testing sets (80% train/20% test)
#     num_train = int(0.8 * len(inputs_by_song))

#     train_inputs, train_labels = inputs_by_song[:num_train], labels_by_song[:num_train]
#     test_inputs, test_labels = inputs_by_song[num_train:], labels_by_song[num_train:]

#     # -- Create training TensorDataset/DataLoader
#     train_dataset = SongDataset(train_inputs, train_labels)
#     train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

#     # -- Create testing TensorDataset/DataLoader
#     test_dataset = SongDataset(test_inputs, test_labels)
#     test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#     return train_dataloader, test_dataloader


def plot_class_counts_hnn(dataloader: DataLoader, model_name: str):
    '''
    Plot class counts for inputs (12 notes) and labels (14 chords) of DataLoader
    '''
    num_input_notes = 12
    num_label_chords = 14

    input_counts = torch.zeros(num_input_notes, dtype=torch.int64)
    label_counts = torch.zeros(num_label_chords, dtype=torch.int64)

    for song_inputs, song_labels in dataloader:

        input_count = torch.sum(song_inputs.squeeze(0), dim=0, dtype=torch.int64)

        label_count = torch.bincount(song_labels.squeeze(0), minlength=num_label_chords)

        input_counts += input_count
        label_counts += label_count

    # -- Plot input note counts
    input_note_classes = [IDX_TO_NOTE_STR_REF.get(note_idx) for note_idx in range(num_input_notes)]
    bars = plt.bar(input_note_classes, input_counts)
    plt.bar_label(bars, padding=3)

    plt.ylim(0, max(input_counts) * 1.1)
    plt.xlabel('Notes')
    plt.ylabel('Count')
    plt.title('Input Class Counts')

    plt.tight_layout()
    plt.savefig(f'saved_models/hnn/{model_name}/figs/input_class_counts.png')
    plt.close()

    # -- Plot label chord counts
    label_chord_classes = [IDX_TO_CHORD_STR_REF.get(chord_idx) for chord_idx in range(num_label_chords)]
    bars = plt.bar(label_chord_classes, label_counts)
    plt.bar_label(bars, padding=3)

    plt.ylim(0, max(label_counts) * 1.1)
    plt.xticks(rotation=45)
    plt.xlabel('Chords')
    plt.ylabel('Count')
    plt.title('Label Class Counts')

    plt.tight_layout()
    plt.savefig(os.path.join(f'saved_models/hnn/{model_name}/figs/label_class_counts.png'))
    plt.close()


def plot_class_counts_mnet(dataloader: DataLoader, model_name: str):
    '''
    Plot class counts for inputs (12 notes) and labels (14 chords) of DataLoader
    '''
    num_input_chords = 84
    num_label_notes = 289

    input_counts = torch.zeros(num_input_chords, dtype=torch.int64)
    label_counts = torch.zeros(num_label_notes, dtype=torch.int64)

    for song_inputs, song_labels in dataloader:
        
        input_count = torch.sum(song_inputs.squeeze(0), dim=0, dtype=torch.int64)

        label_count = torch.bincount(song_labels.squeeze(0), minlength=num_label_notes)

        input_counts += input_count
        label_counts += label_count

    # -- Define possible notes/chord types
    chromatic_notes = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']
    chord_types = ['maj', 'min', 'dim', 'maj7', 'min7', 'dom7', 'min7b5']

    # -- Plot input chord counts
    input_chord_classes = [note + chord_type for note in chromatic_notes for chord_type in chord_types]
    plt.figure(figsize=(20, 10), dpi=300)
    bars = plt.barh(input_chord_classes, input_counts)
    # plt.bar_label(bars, padding=3)

    # plt.ylim(0, max(input_counts) * 1.1)
    # plt.xticks(rotation=45)
    plt.xlabel('Count')
    plt.ylabel('Chords')
    plt.title('Input Class Counts')

    plt.tight_layout()
    plt.savefig(f'saved_models/mnet/{model_name}/figs/input_class_counts.png')
    plt.close()

    # -- Plot label note counts
    label_note_classes = ['rest'] + [note + str(octave) + str(duration) for note in chromatic_notes for octave in range(6) for duration in range(4)]
    plt.figure(figsize=(20, 10), dpi=300)
    bars = plt.bar(label_note_classes[1:1 + 24], label_counts[1:1 + 24])
    plt.bar_label(bars, padding=3)

    plt.ylim(0, max(label_counts) * 1.1)
    plt.xticks(rotation=45)
    plt.xlabel('Notes')
    plt.ylabel('Count')
    plt.title('Label Class Counts')

    plt.tight_layout()
    plt.savefig(os.path.join(f'saved_models/mnet/{model_name}/figs/label_class_counts.png'))
    plt.close()

    # '''
    # Heatmap
    # '''
    # # Assuming note classes are structured as note + octave + duration
    # # Create a DataFrame with multi-level indices
    # notes = ['rest'] + [note + str(octave) for note in chromatic_notes for octave in range(6) for duration in range(4)]
    # df = pd.DataFrame({'Note': label_note_classes, 'Count': label_counts.numpy()})

    # # Example: Pivot table by note and octave
    # # This requires parsing the note names appropriately
    # # Adjust the parsing based on your actual note naming convention

    # # Example parsing (assuming 'C56' where 'C' is note, '5' octave, '6' duration)
    # df['NoteName'] = df['Note'].str.extract(r'([A-G]#?)')
    # df['Octave'] = df['Note'].str.extract(r'[A-G]#?(\d)')
    # df['Duration'] = df['Note'].str.extract(r'[A-G]#?\d(\d)')

    # pivot = df.pivot_table(index='NoteName', columns='Octave', values='Count', aggfunc='sum', fill_value=0)

    # plt.figure(figsize=(20, 10), dpi=300)
    # sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")
    # plt.title('Label Note Counts Heatmap')
    # plt.xlabel('Octave')
    # plt.ylabel('Note')
    # plt.tight_layout()
    # plt.savefig('label_note_counts_heatmap.png')
    # plt.close()



# def create_dataloaders_mnet(subset: int=1):
#     '''
#     Creates training/testing dataloaders for MelodyNet model

#     - chords: One-hot encoding of 84 possible chords (12 notes * 7 types)
#         * [Amaj/min/dim/maj7/min7/dom7/min7b5 -> G#maj/min/dim/maj7/min7/dom7/min7b5]
#         * [0    1   2   3    4    5    6     ... 77    78  79  80   81   82   83    ]
#     - notes: Labels of 145 possible note/octave/lifespan combinations
#         * [rest A2(0) A3(0) ... A6(1) A7(1) ... G#6(1) G#7(1)]
#         * [0    1     2     ... 11    12    ... 143    144   ]
#     '''
#     # -- Parse data into songs list of dicts
#     songs = parse_data(hnn_data=False)

#     # -- Define chords one-hot size 
#     num_chords = 84

#     # -- Initialize note/chord one-hot encoding arrays to index from
#     chord_enc_array = np.eye(num_chords, dtype=int)

#     # -- Initialize empty inputs/labels arrays
#     inputs_by_song = []
#     labels_by_song = []

#     for song in songs:

#         song_inputs = []
#         song_labels = []

#         duration = 0

#         for timestep, (note, chord) in enumerate(zip(song['notes'], song['chords'])):
#             # -- Only take chord input at the beginning of 
#             if duration == 0:
#                 # -- Map chord to one-hot index (7 * note_idx + chord_type_idx)
#                 chord_idx = 7 * NOTE_ENC_TO_IDX_REF.get(chord[:2]) + int(chord[2])
#                 # -- Obtain chord input one-hot encoding
#                 chord_input = chord_enc_array[chord_idx]

#             # -- Map note to class label
#             note_idx = NOTE_ENC_TO_IDX_REF.get(note[:2])
#             note_octave = int(note[2])
#             note_lifespan = int(note[3])

#             # -- On a new note, append the input/label of the previous note
#             if (note_lifespan == 0 or duration == 3) and timestep != 0:
#                 # Rest note
#                 if note_idx is None:
#                     note_label = 0
#                 else:
#                     note_label = 1 + (note_idx * 24) + (note_octave - 2) + (duration * 6)

#                 # -- Append chord to inputs & note to labels
#                 song_inputs.append(chord_input)
#                 song_labels.append(note_label)

#                 duration = 0
#             # -- If the note is sustained, track its duration
#             elif note_lifespan == 1:
#                 duration += 1


#         # print(f"Chord Conversion:\n{[(chord_str, chord_enc) for chord_str, chord_enc in zip(song['chords'], song_inputs)]}\n")
#         # print(f"Note Conversion:\n{[(note_str, note_label) for note_str, note_label in zip(song['notes'], song_labels)]}")

#         # -- Convert song inputs/labels to tensors
#         song_inputs = torch.tensor(np.array(song_inputs, dtype=np.float32))
#         song_labels = torch.tensor(np.array(song_labels, dtype=np.int64))

#         # -- Append song's inputs/labels to full list
#         inputs_by_song.append(song_inputs)
#         labels_by_song.append(song_labels)

#         break

#     # -- Randomly shuffle inputs/labels
#     inputs_and_labels = list(zip(inputs_by_song, labels_by_song))
    
#     random.seed(42)
#     random.shuffle(inputs_and_labels)

#     inputs_by_song, labels_by_song = zip(*inputs_and_labels)

#     # -- Create training/testing sets (80% train/20% test)
#     num_train = int(0.8 * len(inputs_by_song))
#     num_test = len(inputs_by_song) - num_train

#     train_inputs, train_labels = inputs_by_song[:num_train // subset], labels_by_song[:num_train // subset]
#     test_inputs, test_labels = inputs_by_song[num_train // subset:num_train // subset + num_test // subset], \
#                                labels_by_song[num_train // subset:num_train // subset + num_test // subset]

#     # -- Create training TensorDataset/DataLoader
#     train_dataset = SongDataset(train_inputs, train_labels)
#     train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

#     # -- Create testing TensorDataset/DataLoader
#     test_dataset = SongDataset(test_inputs, test_labels)
#     test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#     return train_dataloader, test_dataloader


def create_dataloaders_mnet(subset: int=1, shuffle: bool=True, balance_feature: str=None):
    '''
    New create_dataloaders_mnet

    - Add a chord/melody pair for each note, with the chord's octave and duration
    - During training/testing, the timestep/meter units can be derived from the notes' durations
    '''
    songs = parse_data(hnn_data=False)

    inputs_by_song = []
    labels_by_song = []

    note_duration = 0
    current_note = None
    current_chord = None

    for song in songs:

        song_duration = len(song['notes'])

        chord_inputs = []
        note_labels = []

        for timestep, (note_enc, chord_enc) in enumerate(zip(song['notes'], song['chords'])):
            
            note_lifespan = note_enc[3]

            # First note
            if timestep == 0:
                current_note = note_enc
                current_chord = chord_enc

                # Initialize duration to 1
                note_duration = 1

            else:
                # New note
                if note_lifespan == '0':
                    # Track duration of consecutive rest notes
                    if current_note == '0000' and note_enc == '0000':
                        
                        # Add rest notes on last note
                        if timestep == song_duration - 1:
                            note_duration += 1

                            current_note = current_note[:3] + str(note_duration)

                            note_labels.append(current_note)
                            chord_inputs.append(current_chord)
                            continue
                        # If not the last note, increment duration and continue
                        else:
                            note_duration += 1
                            continue

                    # Update current note's lifespan to duration before storing
                    current_note = current_note[:3] + str(note_duration)

                    note_labels.append(current_note)
                    chord_inputs.append(current_chord)

                    # Update current note/chord to the next note
                    current_note = note_enc
                    current_chord = chord_enc

                    # Initialize duration to 1
                    note_duration = 1

                # Sustained note
                else:
                    note_duration += 1

                # Add current note on last note
                if timestep == song_duration - 1:
                    current_note = current_note[:3] + str(note_duration)

                    note_labels.append(current_note)
                    chord_inputs.append(current_chord)

        inputs_by_song.append(chord_inputs)
        labels_by_song.append(note_labels)


    # -- Convert inputs_by_song/labels_by_song to DataLoaders
    num_chord_classes = 84

    chord_one_hot = np.eye(num_chord_classes, dtype=int)

    input_data = []
    label_data = []

    for song_inputs, song_labels in zip(inputs_by_song, labels_by_song):
        
        song_input_data = []
        song_label_data = []

        for chord_input, note_label in zip(song_inputs, song_labels):
            
            chord_str = CHORD_ENC_TO_STR.get(chord_input)
            chord_idx = CHORD_STR_TO_IDX_MNET.get(chord_str)
            song_input_data.append(chord_one_hot[chord_idx])

            note_str = NOTE_ENC_TO_NOTE_STR_REF.get(note_label[:2]) + note_label[2:]
            note_idx = NOTE_STR_TO_IDX_MNET.get(note_str)
            song_label_data.append(note_idx)

        # Convert song inputs/labels to tensors
        song_input_data = torch.tensor(np.array(song_input_data, dtype=np.float32))
        song_label_data = torch.tensor(np.array(song_label_data, dtype=np.int64))

        # Append to full list of songs
        input_data.append(song_input_data)
        label_data.append(song_label_data)

    # -- Store song keys to evaluate key accuracy during testing
    song_keys = [song['key'] for song in songs]

    # -- Randomly shuffle inputs/labels
    if shuffle:
        inputs_and_labels = list(zip(input_data, label_data, song_keys))

        random.seed(42)
        random.shuffle(inputs_and_labels)

        input_data, label_data, song_keys = zip(*inputs_and_labels)

    # -- Create training/testing sets (80% train/20% test)
    num_train = int(0.8 * len(input_data))
    num_test = len(input_data) - num_train

    train_inputs, train_labels = input_data[:num_train // subset], label_data[:num_train // subset]
    test_inputs, test_labels = input_data[num_train // subset:num_train // subset + num_test // subset], \
                               label_data[num_train // subset:num_train // subset + num_test // subset]
    train_song_keys = song_keys[:num_train // subset]
    test_song_keys = song_keys[num_train // subset:num_train // subset + num_test // subset]

    # -- Create training TensorDataset/DataLoader
    train_dataset = SongDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # -- Create testing TensorDataset/DataLoader
    test_dataset = SongDataset(test_inputs, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if balance_feature:
        train_dataloader = balance_samples(train_dataloader, balance_feature)

    return train_dataloader, test_dataloader, train_song_keys, test_song_keys


def sum_duration(song_labels):
    '''
    Given a song's note class labels, returns the 
    song's total duration
    '''
    song_duration = 0

    note_idx_to_str = {v:k for k,v in NOTE_STR_TO_IDX_MNET.items()}

    for song_label in song_labels:

        # -- Convert song_label to note string
        note_str = note_idx_to_str[song_label.item()]

        # -- Get note duration
        if note_str[1] == '#':
            note_duration = int(note_str[3:])
        else:
            note_duration = int(note_str[2:])

        song_duration += note_duration

    return song_duration

