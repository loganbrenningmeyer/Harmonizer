import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as opt
import numpy as np

import os
import random

'''
-- HNN Model Architecture --

Input Layer:
- 28 total units:
    * 14 State Units
        - Softmax values of previous output
    * 12 Melody Units
        - One-hot encoding of the current melody note (chromatic scale)
    * 2 Meter Units
        - One-hot encoding of the current beat (1st or 3rd beat)

1st Hidden Layer:
- 16 Neurons
    * Connections:
        - Fully connected/learnable with the input layer (feature extractor)

2nd Hidden Layer:
- 12 Neurons (chromatic scale A->G#)
    * Connections:
        - Melody Units
            * Fixed connections between melody units and their 
              corresponding neurons (A input -> A neuron)
            * Learnable connections between melody units and
              other notes' neurons (A input -> A#-G# neurons)
        - 1st Hidden Layer
            * Fully connected/learnable weights with the 1st hidden layer

Output Layer:
- 14 neurons for each chord (7 major, 7 dominant 7th)
    * Connections:
        - 2nd Hidden Layer
            * Partially connected to the 2nd hidden layer
              via fixed weights
                - Establishes appropriate pitch to chord relations
                    * C, E, G ---> Cmaj
        - State Units
            * Recurrently passed back in as the state units of the input
'''

class HNN(nn.Module):
    def __init__(self, hidden1_size: int, lr: float, weight_decay: float,
                       melody_weights: float, chord_weights: float, state_units_decay: float,
                       model_name: str):
        super(HNN, self).__init__()

        # -- Define input sizes
        self.state_size = 14
        self.melody_size = 12
        self.meter_size = 2
        self.input_size = 28

        # -- Define layer sizes
        self.hidden1_size = hidden1_size
        self.hidden2_size = 12
        self.output_size = 14

        # -- Define optimizer parameters
        self.lr = lr
        self.momentum = 0.0
        self.weight_decay = weight_decay

        # -- Define fixed weights and state_units decay rate
        self.melody_weights = melody_weights
        self.chord_weights = chord_weights
        self.state_units_decay = state_units_decay

        # -- Name model
        self.model_name = model_name

        '''
        Define layers/weights
        '''
        # -- 1st Hidden Layer
        self.hidden1 = nn.Linear(self.input_size, self.hidden1_size)

        # -- 2nd Hidden Layer (fully connected/fully learnable: hidden1 -> hidden2)
        self.hidden2_from_hidden1 = nn.Linear(self.hidden1_size, self.hidden2_size) 

        # -- 2nd Hidden Layer (fully connected/partially fixed: melody input -> hidden2)
        self.hidden2_from_melody  = nn.Linear(self.melody_size,  self.hidden2_size, bias=False)

        # Set fixed weights on the diagonal for matching melody notes/hidden2 neurons
        with torch.no_grad():
            self.hidden2_from_melody.weight.fill_diagonal_(self.melody_weights)

        # -- Output layer (partially connected/fixed weights: hidden2 -> output)
        self.output = nn.Linear(self.hidden2_size, self.output_size, bias=False)

        # Define fixed weights for hidden2 notes -> output chords mapping
        notes_to_chords = torch.tensor([
        #    A  A# B  C  C# D  D# E  F  F# G  G#
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],   # Amaj  : A, C#, E
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],   # Bmaj  : B, D#, F#
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],   # Cmaj  : C, E,  G
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],   # Dmaj  : D, F#, A
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],   # Emaj  : E, G#, B
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],   # Fmaj  : F, A,  C
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],   # Gmaj  : G, B,  D
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],   # Adom7 : A, C#, E,  G
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],   # Bdom7 : B, D#, F#, A
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],   # Cdom7 : C, E,  G,  A#
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],   # Ddom7 : D, F#, A,  C
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],   # Edom7 : E, G#, B,  D
            [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],   # Fdom7 : F, A,  C,  D#
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]    # Gdom7 : G, B,  D,  F
        ], dtype=torch.float32)

        # Fixed weights (maj chords = 1/3, dominant 7th chords = 1/4 for balancing note count)
        fixed_output_weights = (notes_to_chords / notes_to_chords.sum(dim=1, keepdim=True)) * self.chord_weights

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

        # melody -> hidden2
        melody = X[:, 14:26]
        h2_from_melody = self.hidden2_from_melody(melody)

        # Combine hidden1 and melody outputs
        h2 = F.relu(h2_from_h1 + h2_from_melody)

        # -- Output Layer
        output = self.output(h2)

        return output
    