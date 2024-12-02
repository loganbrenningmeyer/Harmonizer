import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import os

from collections import Counter

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
    plt.figure(figsize=(20, 10), facecolor=fig_color)
    plt.axes().set_facecolor(fig_color)

    if label_feature != 'classes':
        label_positions = np.arange(len(feature_labels)) * 100
        plt.bar(label_positions, counts_dict.values(), width=75, color=bar_color)
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
    sampler = WeightedRandomSampler(song_weights, num_samples=len(song_weights), replacement=True)

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