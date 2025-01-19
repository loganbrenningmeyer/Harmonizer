import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import Counter

# -- Dissonance Level Distribution Calculations
import torch.distributions as dist
from scipy.stats import entropy

from utils.data.load_data import *
from utils.data.mappings import *

'''
Want to count distributions of classes, notes, octaves, and durations
'''
def count_durations(dataloader: DataLoader):

    duration_counts = {}

    note_idx_to_str = {v:k for k,v in NOTE_STR_TO_IDX_MNET.items()}

    for song_inputs, song_labels in dataloader:

        song_labels = song_labels.squeeze(0)

        for song_label in song_labels:
            
            # -- Convert song_label to note string
            note_str = note_idx_to_str[song_label.item()]

            # -- Get note duration
            if note_str[1] == '#':
                note_duration = int(note_str[3:])
            else:
                note_duration = int(note_str[2:])

            # -- Add duration count
            if note_duration not in duration_counts.keys():
                duration_counts[note_duration] = 1
            else:
                duration_counts[note_duration] += 1

    # -- Sort duration counts
    duration_keys = list(duration_counts.keys())
    duration_keys.sort()

    sorted_duration_counts = {duration: duration_counts[duration] for duration in duration_keys}

    return sorted_duration_counts


def count_samples(dataloader: DataLoader):
    # -- Dictionary of sample counts by each sample feature
    sample_counts = {
        'classes': {},
        'notes': {},
        'octaves': {},
        'notes_octaves': {},
        'durations': {}
    }

    # -- Mapping from note class indices to note strings
    note_idx_to_str = {v:k for k,v in NOTE_STR_TO_IDX_MNET.items()}

    for song_inputs, song_labels in dataloader:
        
        song_labels = song_labels.squeeze(0)

        for song_label in song_labels:

            '''
            Classes
            '''
            label_class = song_label.item()

            sample_counts['classes'][label_class] = sample_counts['classes'].get(label_class, 0) + 1

            '''
            Notes/Octaves/Durations
            '''
            # -- Convert song_label to note string
            note_str = note_idx_to_str[song_label.item()]

            # -- Get label note/octave/duration
            if note_str[1] == '#':
                label_note = note_str[:2]
                label_octave = int(note_str[2])
                label_note_octave = label_note + str(label_octave)
                label_duration = int(note_str[3:])
            else:
                label_note = note_str[:1]
                label_octave = int(note_str[1])
                label_note_octave = label_note + str(label_octave)
                label_duration = int(note_str[2:])

            # -- Count label note
            sample_counts['notes'][label_note] = sample_counts['notes'].get(label_note, 0) + 1

            # -- Count label octave
            sample_counts['octaves'][label_octave] = sample_counts['octaves'].get(label_octave, 0) + 1

            # -- Count label notes/octaves
            sample_counts['notes_octaves'][label_note_octave] = sample_counts['notes_octaves'].get(label_note_octave, 0) + 1

            # -- Count label duration
            sample_counts['durations'][label_duration] = sample_counts['durations'].get(label_duration, 0) + 1

    '''
    Sort dictionaries
    '''
    for sample_feature in sample_counts.keys():

        feature_dict = sample_counts[sample_feature]

        feature_keys = list(feature_dict.keys())
        feature_keys.sort()

        sample_counts[sample_feature] = {feature_key: feature_dict[feature_key] for feature_key in feature_keys}

    return sample_counts


def plot_counts(sample_counts: dict, label_feature: str, balanced: bool, save_dir: str = ''):

    if balanced:
        titles = {
            'classes': 'Balanced Class Labels Distribution',
            'notes': 'Balanced Note Labels Distribution',
            'octaves': 'Balanced Octave Labels Distribution',
            'notes_octaves': 'Balanced Note/Octave Labels Distribution',
            'durations': 'Balanced Duration Labels Distribution'
        }
    else:
        titles = {
            'classes': 'Class Labels Distribution',
            'notes': 'Note Labels Distribution',
            'octaves': 'Octave Labels Distribution',
            'notes_octaves': 'Note/Octave Labels Distribution',
            'durations': 'Duration Labels Distribution'
        }

    counts_dict = sample_counts[label_feature]

    fontname = 'Lato'

    fig_color = '#ffffe6'
    bar_color = '#83a9de'

    # -- Get list of sample feature labels
    feature_labels = counts_dict.keys()

    # -- Create bar plot of sample feature counts
    plt.figure(figsize=(15, 10)) #, facecolor=fig_color)
    # plt.axes().set_facecolor(fig_color)

    if label_feature != 'classes':
        label_positions = np.arange(len(feature_labels)) # * 100
        plt.bar(label_positions, counts_dict.values(), color=bar_color)
    else:
        plt.bar(feature_labels, counts_dict.values(), color=bar_color)

    # -- Adjust plots based on sample_feature
    if label_feature != 'classes':
        if label_feature == 'notes_octaves':
            plt.xticks(ticks=label_positions, labels=feature_labels, rotation=45, fontname=fontname)
        else:
            plt.xticks(ticks=label_positions, labels=feature_labels, fontname=fontname)

    plt.title(titles[label_feature], fontname=fontname, fontsize=18)

    plt.grid(axis='y')
    plt.xlabel(label_feature.capitalize().replace('_', '/'), fontname=fontname, fontweight='bold', fontsize=16)
    plt.ylabel('Count', fontname=fontname, fontweight='bold', fontsize=16)

    plt.xlim((-1, 15.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{label_feature}_distribution.png'), dpi=300)


def balance_samples(dataloader: DataLoader, label_feature: str):
    '''
    Given the specified label feature (classes, notes, octaves, notes/octaves, or durations),
    balances the dataloader to evenly distribute the feature.
    '''
    # -- Determine total counts of each label feature class
    feature_counts = count_samples(dataloader)[label_feature]
    total_samples = sum(feature_counts.values())

    # -- Compute feature frequencies
    feature_frequencies = {feature: count / total_samples for feature, count in feature_counts.items()}

    # -- Compute feature weights (inverse of frequencies)
    feature_weights = {feature: 1.0 / freq for feature, freq in feature_frequencies.items()}

    # -- Normalize feature weights so that the maximum weight is 1
    max_weight = max(feature_weights.values())
    feature_weights = {feature: weight / max_weight for feature, weight in feature_weights.items()}

    # -- Calculate sampling weights for each song
    song_weights = []

    for song_inputs, song_labels in dataloader:

        song_labels = song_labels.squeeze(0)
        feature_counter = Counter()

        # -- Count song's instances of each label feature class
        for song_label in song_labels:
            feature = get_label_feature(song_label.item(), label_feature)
            feature_counter[feature] += 1

        # -- Calculate song weight as the sum of feature weights times their counts
        song_weight = 0.0

        for feature, count in feature_counter.items():
            weight = feature_weights.get(feature, 0)
            song_weight += weight * count

        song_weights.append(song_weight)

    # -- Normalize song weights
    total_weight = sum(song_weights)
    song_weights = [weight / total_weight for weight in song_weights]

    # -- Create WeightedRandomSampler
    generator = torch.Generator()
    generator.manual_seed(42)
    sampler = WeightedRandomSampler(song_weights, 
                                    num_samples=len(song_weights), 
                                    replacement=True,
                                    generator=generator)

    # -- Create new balanced dataloader
    balanced_dataloader = DataLoader(dataloader.dataset, batch_size=1, sampler=sampler)

    return balanced_dataloader


def get_label_feature(label: int, label_feature: str):

    # -- Convert the label to note string
    note_str = NOTE_IDX_TO_STR_MNET.get(label)

    # -- Get label note/octave/duration
    if note_str[1] == '#':
        label_note = note_str[:2]
        label_octave = int(note_str[2])
        label_note_octave = label_note + str(label_octave)
        label_duration = int(note_str[3:])
    else:
        label_note = note_str[:1]
        label_octave = int(note_str[1])
        label_note_octave = label_note + str(label_octave)
        label_duration = int(note_str[2:])

    if label_feature == 'classes':
        return label
    if label_feature == 'notes':
        return label_note
    elif label_feature == 'octaves':
        return label_octave
    elif label_feature == 'notes_octaves':
        return label_note_octave
    elif label_feature == 'durations':
        return label_duration
    

def note_distributions():
    '''
    Returns the average sorted note distributions for 3 levels of dissonance:

    Chord Tones
    - A note is a Chord Tone if it can be found
    within CHORD_TONES[chord]

    Scale Non-Chord Tones
    - A note is a Scale Non-Chord Tone if it is not
    a Chord Tone, but can be found in HARMONIZING_NOTES[chord]
    
    Dissonant Tones
    - A note is a Dissonant Tone if it is neither a
    Chord Tone nor a Scale Non-Chord Tone
            
    
    All distributions have a fixed bin count of 12, achieved by padding missing
    bins with 0s

    Returns:
    - chord_tones, scale_tones, diss_tones: The dataset's average sorted probability
      distributions for each level of dissonance
    '''
    # -- Parse songs data with encoded notes/chords
    songs = parse_data(hnn_data=False)

    num_songs = len(songs)

    # -- Initialize empty total probability distributions
    prob_chord_tones = torch.zeros(12, dtype=torch.float32)
    prob_scale_tones = torch.zeros(12, dtype=torch.float32)
    prob_diss_tones = torch.zeros(12, dtype=torch.float32)

    # -- Count songs without dissonant tones
    no_diss_count = 0

    for song in songs:
        # -- Extract song's notes and chords
        notes = song['notes']
        chords = song['chords']

        # -- Establish note to histogram bin mapping
        note_indices = {
            'A' : 0,
            'A#': 1,
            'B' : 2,
            'C' : 3,
            'C#': 4,
            'D' : 5,
            'D#': 6,
            'E' : 7,
            'F' : 8,
            'F#': 9,
            'G' : 10,
            'G#': 11
        }

        # -- Initialize empty histograms
        chord_tones = torch.zeros(12, dtype=torch.float32)
        scale_tones = torch.zeros(12, dtype=torch.float32)
        diss_tones = torch.zeros(12, dtype=torch.float32)

        #-- For each note, chord: Add count for level of dissonance
        for note, chord in zip(notes, chords):

            # Convert note/chord to strings
            note = NOTE_ENC_TO_NOTE_STR_REF.get(note[:2])
            chord = CHORD_ENC_TO_STR.get(chord)

            # Ignore rests
            if note == 'R':
                continue

            # Determine note index
            note_idx = note_indices[note]

            # -- Chord Tones
            if note in CHORD_TONES.get(chord):
                chord_tones[note_idx] += 1
            
            # -- Scale Non-Chord Tones
            elif note in HARMONIZING_NOTES.get(chord):
                scale_tones[note_idx] += 1

            # -- Dissonant Tones
            else:
                diss_tones[note_idx] += 1

        # -- Sort histograms
        chord_tones = chord_tones.sort(descending=True)[0]
        scale_tones = scale_tones.sort(descending=True)[0]
        diss_tones = diss_tones.sort(descending=True)[0]

        # -- Convert to probability distributions
        total_chord_tones = torch.sum(chord_tones)
        total_scale_tones = torch.sum(scale_tones)
        total_diss_tones = torch.sum(diss_tones)

        chord_tones /= total_chord_tones
        scale_tones /= total_scale_tones
        if total_diss_tones != 0:
            diss_tones /= total_diss_tones
        else:
            no_diss_count += 1

        # -- Add to total dataset probability distributions
        prob_chord_tones += chord_tones
        prob_scale_tones += scale_tones
        prob_diss_tones += diss_tones

    # -- Compute the average probability distributions
    prob_chord_tones /= num_songs
    prob_scale_tones /= num_songs
    prob_diss_tones /= (num_songs - no_diss_count)

    # -- Ensure probability distributions sum to 1
    prob_chord_tones /= torch.sum(prob_chord_tones)
    prob_scale_tones /= torch.sum(prob_scale_tones)
    prob_diss_tones /= torch.sum(prob_diss_tones)

    return prob_chord_tones, prob_scale_tones, prob_diss_tones


def save_distributions():
    chord_tones, scale_tones, diss_tones = note_distributions()

    # -- Plot distributions of levels of dissonance
    chord_tones = chord_tones.cpu().numpy()
    scale_tones = scale_tones.cpu().numpy()
    diss_tones = diss_tones.cpu().numpy()

    x = np.arange(12)

    fig, axs = plt.subplots(3, figsize=(8, 10))

    axs[0].bar(x, chord_tones)
    axs[1].bar(x, scale_tones)
    axs[2].bar(x, diss_tones)

    axs[0].set_title('Chord Tones')
    axs[1].set_title('Scale Non-Chord Tones')
    axs[2].set_title('Dissonant Tones')

    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[2].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig('dataset_dists.png')

    # -- Compute entropy of each distribution
    chord_tones_H = entropy(chord_tones)
    scale_tones_H = entropy(scale_tones)
    diss_tones_H = entropy(diss_tones)

    print(f"H(chord): {chord_tones_H}, H(scale): {scale_tones_H}, H(diss): {diss_tones_H}")

    # -- Write distribution/entropy information to csvs
    dists = {
        'Chord': chord_tones,
        'Scale': scale_tones,
        'Dissonant': diss_tones,
    }

    df = pd.DataFrame(dists)
    df.to_csv('dataset_dists.csv', index=False)

    entropies = {
        'Chord': [chord_tones_H],
        'Scale': [scale_tones_H],
        'Dissonant': [diss_tones_H]
    }

    df = pd.DataFrame(entropies)
    df.to_csv('dataset_entropies.csv', index=False)    



