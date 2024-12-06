import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as opt
import numpy as np

import os
import random

from utils.data.mappings import *

'''
-- MelodyNet Model Architecture --

Classes:
- Chords: Root notes of the chromatic scale in 7 chord types
    * 12 notes * 7 chords = 84 chord classes
- Melody: Predicts [note-accidental-octave-lifespan]
    * 12 notes * 6 octaves * 4 durations + rest class = 289 classes
- Meter: 16th note beat in bar
    * 16 classes

Input Layer:
- 289 total units (+ noise units if inject_noise == True):
    * 289 State Units:
        - Melody class softmax values of previous output
    * 84 Chord Units:
        - One-hot encoding of the current chord
    * 16 Meter Units:
        - One-hot encoding of the current 16th note beat
    * noise_size Noise Units:
        - Length noise_size array of random normal values [0, 1]

1st Hidden Layer:
- ? Neurons
    * Connections:
        - Fully connected/learnable with the input layer (feature extractor)

2nd Hidden Layer:
- 84 Neurons
    * Connections:
        - Chord Units
            * Fixed connections between chord units and their
              corresponding neurons (Cmaj -> Cmaj neuron)
            * Learnable connections between chord units and 
              other chords' neurons (Cmaj -> 83 other chords)
        - 1st Hidden Layer
            * Fully connected/learnable weights

Output Layer:
- 289 Neurons (for each melody note class)
    * Connections:
        - 2nd Hidden Layer
            * Partially connected to the 2nd hidden layer 
              via fixed weights
                - Establishes appropriate chord to melody note relations
                    * Cmaj ---> C, E, G
        - State Units
            * Recurrently passed back as the state units of the input
'''

# class MelodyNet(nn.Module):
#     def __init__(self, hidden1_size: int, lr: float, weight_decay: float, repetition_weight: float,
#                        chord_weight: float, melody_weight: float, state_units_decay: float, rest_weight: float,
#                        model_name: str):
#         super(MelodyNet, self).__init__()

#         # -- Define input sizes
#         self.state_size = 145
#         self.chord_size = 84
#         self.meter_size = 16
#         self.input_size = 245

#         # -- Define layer sizes
#         self.hidden1_size = hidden1_size
#         self.hidden2_size = 84
#         self.output_size = 145

#         # -- Define repetition penalty weight
#         self.repetition_weight = repetition_weight

#         # -- Define optimizer parameters
#         self.lr = lr 
#         self.momentum = 0.0
#         self.weight_decay = weight_decay

#         # -- Define fixed weights, state_units decay rate, and rest notes weight
#         self.chord_weight = chord_weight
#         self.melody_weight = melody_weight
#         self.state_units_decay = state_units_decay
#         self.rest_weight = rest_weight

#         # -- Name model
#         self.model_name = model_name

#         '''
#         Define layers/weights
#         '''
#         # -- 1st Hidden Layer
#         self.hidden1 = nn.Linear(self.input_size, self.hidden1_size)

#         # -- 2nd Hidden Layer (fully connected/fully learnable: hidden1 -> hidden2)
#         self.hidden2_from_hidden1 = nn.Linear(self.hidden1_size, self.hidden2_size)

#         # -- 2nd Hidden Layer (fully connected/partially fixed: chord input -> hidden2)
#         self.hidden2_from_chord   = nn.Linear(self.chord_size,   self.hidden2_size, bias=False)

#         # Set fixed weights on the diagonal for matching chords/hidden2 neurons
#         with torch.no_grad():
#             self.hidden2_from_chord.weight.fill_diagonal_(self.chord_weight)

#         # -- Output layer (partially connected/fixed weights: hidden2 -> output)
#         self.output = nn.Linear(self.hidden2_size, self.output_size, bias=False)

#         '''
#         Define fixed weights for hidden2 chords -> output melody notes mapping

#         # Input Chords (84) →
#         [                      ] # Output Notes (145) ↓
#         [                      ]
#         [                      ]
#         [                      ]
#         [                      ]
#         [                      ]
#         [                      ]
#         [                      ]
#         '''
#         # Define input chords array and output notes array
#         chromatic_notes = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']
#         chord_types = ['maj', 'min', 'dim', 'maj7', 'min7', 'dom7', 'min7b5']
        
#         # Amaj/min/dim/maj7/min7/dom7/min7b5 -> G#maj/min/dim/maj7/min7/dom7/min7b5 (84)
#         hidden2_chords = [note + chord_type for note in chromatic_notes for chord_type in chord_types]
        
#         # rest (0000) + [A2-A7 (0) + A2-A7 (1)] -> [G#2-G#7 (0) + G#2-G#7 (1)] (145)
#         output_notes = ['rest'] + [note for note in chromatic_notes for octave in range(6) for lifespan in range(2)]

#         # Initialize empty fixed weights array of chords to notes mappings
#         chords_to_notes = torch.zeros((self.output_size, self.hidden2_size))

#         # Map all chords to rest note (reduced weights)
#         chords_to_notes[0] = torch.ones(self.hidden2_size)
        
#         # Use CHORD_NOTES mappings to automatically fill in the fixed weights array
#         for i, note in enumerate(output_notes[1:], start=1):
#             for j, chord in enumerate(hidden2_chords):
#                 # Get chord notes
#                 chord_notes = CHORD_NOTES.get(chord)
#                 # If output_note in input_chord, set fixed weight to 1
#                 if any(note == chord_note[:-1] for chord_note in chord_notes):
#                     chords_to_notes[i][j] = 1

#         # Balance and scale fixed weights
#         chords_to_notes = (chords_to_notes / chords_to_notes.sum(dim=1, keepdim=True)) * self.melody_weight

#         # Reduce scale rest note weights
#         chords_to_notes[0] *= self.rest_weight
            
#         # Balance and scale fixed weights
#         fixed_output_weights = chords_to_notes

#         with torch.no_grad():
#             self.output.weight.copy_(fixed_output_weights)

#         # Ensure that fixed output weights do not update
#         self.output.weight.requires_grad = False


#     def forward(self, X):
#         # -- 1st Hidden Layer
#         h1 = F.relu(self.hidden1(X))

#         # -- 2nd Hidden Layer
#         # hidden1 -> hidden2
#         h2_from_h1 = self.hidden2_from_hidden1(h1)
        
#         # chord -> hidden2
#         chord = X[:, self.state_size : self.state_size + self.chord_size]
#         h2_from_chord = self.hidden2_from_chord(chord)

#         # Combine hidden1 and chord outputs
#         h2 = F.relu(h2_from_h1 + h2_from_chord)

#         # -- Output Layer
#         output = self.output(h2)

#         return output
    

class MelodyNet(nn.Module):
    def __init__(self, hidden1_size: int, lr: float, weight_decay: float, 
                       repetition_loss: float, key_loss: float, harmony_loss: float,
                       chord_weight: float, melody_weight: float, state_units_decay: float,
                       fixed_chords: bool, fixed_melody: bool,
                       rest_fixed_weight: float, rest_loss_weight: float,
                       inject_noise: bool, noise_size: int, noise_weight: float,
                       temperature: float, dropout_rate: float,
                       model_name: str):
        super(MelodyNet, self).__init__()

        '''
        Output class:
        - 459 total classes (note, octave, duration combinations in the dataset)

        Input class:
        - state units (459) + chord units (84) + meter units (16) = 559
        '''
        # -- Define noise params
        self.inject_noise = inject_noise
        if inject_noise:
            self.noise_size = noise_size
            self.noise_weight = noise_weight
        else:
            self.noise_size = 0
            self.noise_weight = 0.0

        # -- Define input sizes
        self.melody_size = 459                  # number of output melody note classes
        self.chord_size = 84                    # number of input chord classes
        self.meter_size = 16                    # number of timesteps in a measure
        self.state_size = self.melody_size      # number of state units (same as output)
        self.input_size = self.melody_size + self.chord_size + self.meter_size + self.noise_size

        # -- Define layer sizes
        self.hidden1_size = hidden1_size
        self.hidden2_size = 84
        self.output_size = self.melody_size

        # -- Define loss penalties
        self.repetition_loss = repetition_loss
        self.key_loss = key_loss
        self.harmony_loss = harmony_loss

        # -- Define optimizer parameters
        self.lr = lr 
        self.momentum = 0.0
        self.weight_decay = weight_decay

        # -- Define fixed weights, state_units decay rate, and rest fixed/loss weights
        self.chord_weight = chord_weight
        self.melody_weight = melody_weight
        self.fixed_chords = fixed_chords
        self.fixed_melody = fixed_melody
        self.state_units_decay = state_units_decay
        self.rest_fixed_weight = rest_fixed_weight
        self.rest_loss_weight = rest_loss_weight
        

        # -- Define "pro-creativity" parameters
        self.temperature = temperature
        self.dropout_rate = dropout_rate

        # -- Name model
        self.model_name = model_name

        '''
        Define layers/weights
        '''
        # -- 1st Hidden Layer
        self.hidden1 = nn.Linear(self.input_size, hidden1_size)

        # -- Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # -- 2nd Hidden Layer (fully connected/fully learnable: hidden1 -> hidden2)
        self.hidden2_from_hidden1 = nn.Linear(hidden1_size, self.hidden2_size)

        # -- 2nd Hidden Layer (fully connected/partially fixed: chord input -> hidden2)
        self.hidden2_from_chord   = nn.Linear(self.chord_size,   self.hidden2_size, bias=False)

        # Set fixed weights on the diagonal for matching chords/hidden2 neurons
        if fixed_chords:
            with torch.no_grad():
                self.hidden2_from_chord.weight.fill_diagonal_(chord_weight)

        # -- Output layer (partially connected/fixed weights: hidden2 -> output)
        self.output_fixed = nn.Linear(self.hidden2_size, self.output_size, bias=False)
        self.output_learnable = nn.Linear(self.hidden2_size, self.output_size, bias=True)

        '''''''''''''''''''''''''''''''''


        
        
        Fixed weights for h2 chord to
        output melody note mapping:

        Input Chords (84) →
        [                      ] Output 
        [                      ] Notes 
        [                      ] (459) ↓
        [                      ]
        [                      ]
        [                      ]
        [                      ]
        [                      ]
        [                      ]




        

        '''''''''''''''''''''''''''''''''

        # Initialize empty fixed weights array of chords to notes mappings
        output_fixed_weights = torch.zeros((self.output_size, self.hidden2_size))

        if fixed_melody:
            for note_str, note_idx in NOTE_STR_TO_IDX_MNET.items():
                for chord_str, chord_idx in CHORD_STR_TO_IDX_MNET.items():
                    # Get constituent chord notes
                    chord_notes = CHORD_NOTES.get(chord_str)
                    # Get note from note_str
                    if note_str[1] == '#':
                        note = note_str[:2]
                    else:
                        note = note_str[0]

                    # Set weight to 1 if the chord contains the note or the note is a rest note
                    if any(note == chord_note[:-1] for chord_note in chord_notes):
                        output_fixed_weights[note_idx][chord_idx] = 1
                    
            # Temporarily set rest note weights to 1 to avoid divide by 0
            output_fixed_weights[0:41] = torch.ones((41, self.chord_size))

            # Balance and scale fixed weights
            output_fixed_weights = (output_fixed_weights / output_fixed_weights.sum(dim=1, keepdim=True)) * melody_weight

            # Reduce scale rest note weights (rows 0-40)
            output_fixed_weights[0:41] *= rest_fixed_weight

        with torch.no_grad():
            self.output_fixed.weight.copy_(output_fixed_weights)
            self.output_fixed.weight.requires_grad = False


    def forward(self, X):
        # -- 1st Hidden Layer
        h1 = F.relu(self.hidden1(X))

        # -- Dropout
        h1 = self.dropout(h1)

        # -- 2nd Hidden Layer
        # hidden1 -> hidden2
        h2_from_h1 = self.hidden2_from_hidden1(h1)
        
        # chord -> hidden2
        chord = X[:, self.state_size : self.state_size + self.chord_size]
        h2_from_chord = self.hidden2_from_chord(chord)

        # Combine hidden1 and chord outputs
        h2 = F.relu(h2_from_h1 + h2_from_chord)

        # -- Output Layer
        output_fixed = self.output_fixed(h2)
        output_learnable = self.output_learnable(h2)
        output = output_fixed + output_learnable

        return output
    
