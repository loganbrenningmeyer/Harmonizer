import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as opt
import numpy as np

import os
import random

'''
-- Model Architecture --

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
        - 2nd hidden layer
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
        notes_to_chord = torch.tensor([
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
        fixed_output_weights = (notes_to_chord / notes_to_chord.sum(dim=1, keepdim=True)) * self.chord_weights

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


# def train(model: HNN, dataloader: DataLoader, criterion: nn.Module, optimizer: opt.Optimizer, device: torch.device):
#     # -- Set model to train
#     model.train()

#     total_loss = 0.0

#     # -- Iterate through DataLoader batches (each batch is a song)
#     for song_inputs, song_labels in dataloader:
#         # -- Put song data onto device and add batch dim
#         song_inputs = song_inputs.squeeze(0).to(device)  # Shape: [sequence_length, input_size]
#         song_labels = song_labels.squeeze(0).to(device)  # Shape: [sequence_length]

#         # -- Initialize state units to 0 (will update w/ outputs in loop)
#         state_units = torch.zeros((1, model.output_size)).to(device)

#         # -- Accumulate loss per song
#         song_loss = 0.0

#         # -- Iterate through each song timestep (each 1st/3rd beat)
#         for timestep in range(song_inputs.size(0)):
#             # Get melody input/chord label for current timestep
#             input_t = song_inputs[timestep].unsqueeze(0)
#             label_t = song_labels[timestep].unsqueeze(0)

#             # Define meter_units based on batch index
#             meter_units = F.one_hot(torch.arange(2, dtype=torch.long))[timestep % 2].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
#             meter_units = meter_units.expand((dataloader.batch_size, 2))                        # Create batch dimension

#             # Concatenate state_units, melody inputs, and meter_units
#             inputs = torch.cat([state_units, input_t, meter_units], dim=1)

#             # Zero parameter gradients
#             optimizer.zero_grad()
#             # Forward pass
#             output = model(inputs)
#             # Compute loss
#             loss = criterion(output, label_t)
#             song_loss += loss.item()
#             # Backward pass
#             loss.backward()

#             # Zero fixed weight gradients
#             model.hidden2_from_melody.weight.grad.fill_diagonal_(0.0)

#             # Update weights
#             optimizer.step()

#             # Update state units w/ outputs softmax and normalize
#             state_units = F.softmax(output.detach(), dim=1).to(device) + model.state_units_decay * state_units
#             state_units = state_units / state_units.sum(dim=1, keepdim=True)

#         # Add batch loss to total loss
#         total_loss += song_loss

#     # -- Compute average epoch loss
#     epoch_loss = total_loss / len(dataloader)

#     return epoch_loss


# def test(model: HNN, dataloader: DataLoader, device: torch.device):
#     # -- Set model to eval mode
#     model.eval()

#     num_correct = 0

#     # -- Iterate through testing DataLoader
#     for song_inputs, song_labels in dataloader:
#         # -- Put song data onto device and add batch dim
#         song_inputs = song_inputs.squeeze(0).to(device)  # Shape: [sequence_length, input_size]
#         song_labels = song_labels.squeeze(0).to(device)  # Shape: [sequence_length]

#         # -- Initialize state units to 0 (will update w/ outputs in loop)
#         state_units = torch.zeros((1, model.output_size)).to(device)

#         # -- Iterate through each song timestep (each 1st/3rd beat)
#         for timestep in range(song_inputs.size(0)):
#             # Get melody input/chord label for current timestep
#             input_t = song_inputs[timestep].unsqueeze(0)
#             label_t = song_labels[timestep].unsqueeze(0)

#             # Define meter_units based on batch index
#             meter_units = F.one_hot(torch.arange(2, dtype=torch.long))[timestep % 2].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
#             meter_units = meter_units.expand((dataloader.batch_size, 2))                        # Create batch dimension

#             # Concatenate state_units, melody inputs, and meter_units
#             inputs = torch.cat([state_units, input_t, meter_units], dim=1)

#             # print(f"inputs: {inputs}")

#             # Forward pass
#             output = model(inputs)

#             # Determine if chord prediction is correct
#             # print(f"label_t: {label_t}, output: {output}")

#             label_idx = label_t.item()
#             chord_idx = torch.argmax(output.detach().squeeze(0)).item()

#             # print(f"label_idx: {label_idx}, chord_idx: {chord_idx}")

#             if chord_idx == label_t:
#                 num_correct += 1

#             # Update state units
#             state_units = F.softmax(output, dim=1) + model.state_units_decay * state_units
#             state_units = state_units / state_units.sum(dim=1, keepdim=True)

#     # -- Compute testing accuracy
#     total_samples = sum([len(song_inputs[0]) for song_inputs, _ in dataloader])
#     accuracy = num_correct / total_samples

#     return accuracy


# def main():
#     '''
#     Prepare Data
#     '''
#     # -- Determine device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # -- Get train/test inputs/labels
#     train_inputs, train_labels, test_inputs, test_labels = load_data.create_data()

#     # -- Create training TensorDataset/DataLoader
#     train_dataset = SongDataset(train_inputs, train_labels)
#     train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

#     # -- Create testing TensorDataset/DataLoader
#     test_dataset = SongDataset(test_inputs, test_labels)
#     test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#     '''
#     Create Model
#     '''
#     # -- Initialize model
#     hnn = HNN().to(device)

#     # -- Define criterion/optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = opt.SGD(hnn.parameters(), lr=hnn.lr, momentum=hnn.momentum, weight_decay=hnn.weight_decay)

#     '''
#     Training/Testing
#     '''
#     # -- Create model saving directory
#     os.makedirs(f'hnn_models/{hnn.model_name}', exist_ok=True)

#     epochs = 10
#     for epoch in range(1, epochs + 1):
#         epoch_loss = train(hnn, train_dataloader, criterion, optimizer, device)
#         print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

#         if epoch % 10 == 0:
#             # -- Save trained model
#             torch.save(hnn, f'hnn_models/{hnn.model_name}/epoch{epoch}.pth')

#     '''
#     Testing
#     '''
#     accuracy = test(hnn, test_dataloader, device)

#     print(f"Testing Accuracy: {accuracy*100:.2f}%")


# if __name__ == "__main__":
#     main()