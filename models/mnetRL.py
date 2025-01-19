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
    

class MelodyNetRL(nn.Module):
    def __init__(self, hidden1_size: int, lr: float, weight_decay: float, 
                       chord_weight: float, melody_weight: float, rest_fixed_weight: float,
                       fixed_chords: bool, fixed_melody: bool, dropout_rate: float,
                       reward_weight: float, heuristic_params: dict,
                       model_name: str):
        super(MelodyNetRL, self).__init__()

        '''
        Output class:
        - 459 total classes (note, octave, duration combinations in the dataset)

        Input class:
        - state units (459) + chord units (84) + meter units (16) = 559
        '''
        # -- Define noise params
        # self.inject_noise = inject_noise
        # if inject_noise:
        #     self.noise_size = noise_size
        #     self.noise_weight = noise_weight
        # else:
        #     self.noise_size = 0
        #     self.noise_weight = 0.0

        # -- Define input sizes
        self.melody_size = 459                  # number of output melody note classes
        self.chord_size = 84                    # number of input chord classes
        self.meter_size = 16                    # number of timesteps in a measure
        self.state_size = 256                   # number of state units (8 * 32 size feature embeddings)
        # self.state_size = self.melody_size
        self.input_size = self.state_size + self.chord_size + self.meter_size

        # -- Initialize State Units Feature Embeddings
        self.n_past_notes = 8
        self.vocab_size = self.melody_size      # (459 classes, indices [0, 458])
        self.embedding_dim = 32
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size + 1,     # add 1 to include the 'no-note' token (idx 459)
                                            embedding_dim=self.embedding_dim)
        class_indices = torch.arange(0, self.melody_size)
        self.class_vectors = self.embedding_layer(class_indices)

        # -- Define layer sizes
        self.hidden1_size = hidden1_size
        self.hidden2_size = 84
        self.output_size = self.melody_size

        # -- Define loss penalties
        # self.repetition_loss = repetition_loss
        # self.key_loss = key_loss
        # self.harmony_loss = harmony_loss

        # -- Define optimizer parameters
        self.lr = lr 
        self.momentum = 0.0
        self.weight_decay = weight_decay

        # self.state_units_decay = state_units_decay

        # -- Define fixed weights, state_units decay rate, and rest fixed/loss weights
        self.chord_weight = chord_weight
        self.melody_weight = melody_weight
        self.fixed_chords = fixed_chords
        self.fixed_melody = fixed_melody
        # self.state_units_decay = state_units_decay
        self.rest_fixed_weight = rest_fixed_weight
        # self.rest_loss_weight = rest_loss_weight

        self.reward_weight = reward_weight
        self.heuristic_params = heuristic_params
        

        # -- Define "pro-creativity" parameters
        # self.temperature = temperature
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
    
