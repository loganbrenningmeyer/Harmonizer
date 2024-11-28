import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as opt
import numpy as np

import os
import random

'''
-- MelodyNet Model Architecture --

Classes:
- Chords: Root notes of the chromatic scale in 7 chord types
    * 12 notes * 7 chords = 84 chord classes
- Melody: Predicts [note-accidental-octave-lifespan]
    * 12 notes * 6 octaves * 2 lifespans + rest class = 145 classes
- Meter: 16th note beat in bar
    * 16 classes

Input Layer:
- 245 total units:
    * 145 State Units:
        - Melody class softmax values of previous output
    * 84 Chord Units:
        - One-hot encoding of the current chord
    * 16 Meter Units:
        - One-hot encoding of the current 16th note beat

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
- 145 Neurons (for each melody note class)
    * Connections:
        - 2nd Hidden Layer
            * Partially connected to the 2nd hidden layer 
              via fixed weights
                - Establishes appropriate chord to melody note relations
                    * Cmaj ---> C, E, G
        - State Units
            * Recurrently passed back as the state units of the input

!! Need to figure out if there all 7 octaves are in the data !!
'''

class MelodyNet(nn.Module):
    def __init__(self, hidden1_size: int,
                       chord_weights: float, melody_weights: float,
                       model_name: str):
        super(MelodyNet, self).__init__()

        # -- Define input sizes
        self.state_size = 145
        self.chord_size = 84
        self.meter_size = 16
        self.input_size = 245

        # -- Define layer sizes
        self.hidden1_size = hidden1_size
        self.hidden2_size = 84
        self.output_size = 145

        # -- Define optimizer parameters

        # -- Define fixed weights and state_units decay rate
        self.chord_weights = chord_weights
        self.melody_weights = melody_weights

        # -- Name model
        self.model_name = model_name

        '''
        Define layers/weights
        '''
        # -- 1st Hidden Layer
        self.hidden1 = nn.Linear(self.input_size, self.hidden1_size)

        # -- 2nd Hidden Layer (fully connected/fully learnable: hidden1 -> hidden2)
        self.hidden2_from_hidden1 = nn.Linear(self.hidden1_size, self.hidden2_size)

        # -- 2nd Hidden Layer (fully connected/partially fixed: chord input -> hidden2)
        self.hidden2_from_chord   = nn.Linear(self.chord_size,   self.hidden2_size, bias=False)

        # Set fixed weights on the diagonal for matching chords/hidden2 neurons
        with torch.no_grad():
            self.hidden2_from_chord.weight.fill_diagonal_(self.chord_weights)

        # -- Output layer (partially connected/fixed weights: hidden2 -> output)
        self.output = nn.Linear(self.hidden2_size, self.output_size, bias=False)

        # Define fixed weights for hidden2 chords -> output melody notes mapping
        '''
        # Input Chords (84) →
        [                      ] # Output Notes (145) ↓
        [                      ]
        [                      ]
        [                      ]
        [                      ]
        [                      ]
        [                      ]
        [                      ]

        Chords:
        - Amaj -> G#maj
        - Amin -> G#min
        - Adim -> G#dim
        - Amaj7 -> G#maj7
        - Amin7 -> G#min7
        - Adom7 -> G#dom7
        - Amin7b5 -> G#min7b5

        Notes:
        - Rest
        - A0-6
        - A#0-6
        - B0-6
        - ⁝
        - G#0-6

        Use mappings.py to automatically fill in the fixed weights array
        '''
        chords_to_notes = torch.tensor([
            
        ])

        # Balance and scale fixed weights
        fixed_output_weights = (chords_to_notes / chords_to_notes.sum(dim=1, keepdim=True)) * self.melody_weights

        with torch.no_grad():
            self.output.weight.copy_(fixed_output_weights)

        # Ensure that fixed output weights do not update
        self.output.weight.requires_grad = False

    def forward(self, X):
        # -- 1st Hidden Layer
        h1 = F.relu(self.hidden1(X))

        # -- 2nd Hidden Layer
        # hidden1 -> hidden2
        h2_from_h1 = self.hidden2_from_hidden1(h1)
        
        # chord -> hidden2
        chord = X[:, ]      # Figure out number of classes first
        h2_from_chord = self.hidden2_from_chord(chord)

        # Combine hidden1 and chord outputs
        h2 = F.relu(h2_from_h1 + h2_from_chord)

        # -- Output Layer
        output = self.output(h2)

        return output