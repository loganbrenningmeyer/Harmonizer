import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torch.optim as opt

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
    def __init__(self):
        super(HNN, self).__init__()

        # -- Define input sizes
        self.state_size = 14
        self.melody_size = 12
        self.meter_size = 2
        self.input_size = 28

        # -- Define layer sizes
        self.hidden1_size = 16
        self.hidden2_size = 12
        self.output_size = 14

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
            self.hidden2_from_melody.weight.fill_diagonal_(1.0)

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
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],   # A7    : A, C#, E,  G
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],   # B7    : B, D#, F#, A
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],   # C7    : C, E,  G,  Bb
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],   # D7    : D, F#, A,  C
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],   # E7    : E, G#, B,  D
            [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],   # F7    : F, A,  C,  Eb
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]    # G7    : G, B,  D,  F
        ], dtype=torch.float32)

        # Fixed weights (maj chords = 1/3, dominant 7th chords = 1/4 for balancing note count)
        fixed_output_weights = notes_to_chord / notes_to_chord.sum(dim=1, keepdim=True)

        with torch.no_grad():
            self.output.weight.copy_(fixed_output_weights)

        # Ensure that fixed output weights do not update
        self.output.weight.requires_grad = False

        # -- State Units (output layer softmax)
        self.state_units = torch.zeros(size=self.output_size)

    
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


def train(model: HNN, dataloader: DataLoader, criterion: nn.Module, optimizer: opt.Optimizer, device: torch.device):
    # -- Set model to train
    model.train()

    total_loss = 0.0

    # -- Initialize state units to 0 (will update w/ outputs in loop)
    state_units = torch.zeros(dataloader.batch_size, model.output_size)

    # -- Iterate through DataLoader batches/song beats (melody unit inputs/chord outputs)
    for i, (inputs, labels) in enumerate(dataloader):
        # Define meter_units based on batch index
        meter_units = F.one_hot(torch.arange(2, dtype=torch.float32))[i % 2].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
        meter_units = meter_units.expand((dataloader.batch_size, 2))                        # Create batch dimension

        # Concatenate state_units, melody inputs, and meter_units
        inputs = torch.cat([state_units, inputs, meter_units], dim=1)
        
        # Send inputs/labels to torch device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()

        # Zero fixed weight gradients
        with torch.no_grad():
            model.hidden2_from_melody.weight.grad.fill_diagonal_(0.0)

        # Update weights
        optimizer.step()

        # Update state units w/ outputs softmax
        state_units = F.softmax(outputs.detach(), dim=1).to(device)

        # Add batch loss to total loss
        total_loss += loss.item() * inputs.size(0)

    # -- Compute average epoch loss
    epoch_loss = total_loss / len(dataloader.dataset)

    return epoch_loss

def main():
    # -- Initialize model
    hnn = HNN()

    # -- Create DataLoader
    dataloader = DataLoader()

    # -- Define criterion/optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.SGD(hnn.parameters(), lr=0.01, momentum=0.0)

    # -- Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Train Model
    train(hnn, dataloader, criterion, optimizer, device)

    # -- Test Model

if __name__ == "__main__":
    main()