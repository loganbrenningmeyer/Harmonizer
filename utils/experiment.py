import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader

from models.hnn import HNN
from models.mnet import MelodyNet


def train_mnet(model: MelodyNet, dataloader: DataLoader, criterion: nn.Module, optimizer: opt.Optimizer, device: torch.device):
    # -- Set model to train
    model.train()

    


def train(model: HNN, dataloader: DataLoader, criterion: nn.Module, optimizer: opt.Optimizer, device: torch.device):
    # -- Set model to train
    model.train()

    total_loss = 0.0

    # -- Iterate through DataLoader batches (each batch is a song)
    for song_inputs, song_labels in dataloader:
        # -- Put song data onto device and add batch dim
        song_inputs = song_inputs.squeeze(0).to(device)  # Shape: [sequence_length, input_size]
        song_labels = song_labels.squeeze(0).to(device)  # Shape: [sequence_length]

        # -- Initialize state units to 0 (will update w/ outputs in loop)
        state_units = torch.zeros((1, model.output_size)).to(device)

        # -- Accumulate loss per song
        song_loss = 0.0

        # -- Iterate through each song timestep (each 1st/3rd beat)
        for timestep in range(song_inputs.size(0)):
            # Get melody input/chord label for current timestep
            input_t = song_inputs[timestep].unsqueeze(0)
            label_t = song_labels[timestep].unsqueeze(0)

            # Define meter_units based on batch index
            meter_units = F.one_hot(torch.arange(2, dtype=torch.long))[timestep % 2].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
            meter_units = meter_units.expand((dataloader.batch_size, 2))                        # Create batch dimension

            # Concatenate state_units, melody inputs, and meter_units
            inputs = torch.cat([state_units, input_t, meter_units], dim=1)

            # Zero parameter gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(inputs)
            # Compute loss
            loss = criterion(output, label_t)
            song_loss += loss.item()
            # Backward pass
            loss.backward()

            # Zero fixed weight gradients
            model.hidden2_from_melody.weight.grad.fill_diagonal_(0.0)

            # Update weights
            optimizer.step()

            # Update state units w/ outputs softmax and normalize
            state_units = F.softmax(output.detach(), dim=1).to(device) + model.state_units_decay * state_units
            state_units = state_units / state_units.sum(dim=1, keepdim=True)

        # Add batch loss to total loss
        total_loss += song_loss

    # -- Compute average epoch loss
    epoch_loss = total_loss / len(dataloader)

    return epoch_loss


def test(model: HNN, dataloader: DataLoader, device: torch.device):
    # -- Set model to eval mode
    model.eval()

    num_correct = 0

    # -- Iterate through testing DataLoader
    for song_inputs, song_labels in dataloader:
        # -- Put song data onto device and add batch dim
        song_inputs = song_inputs.squeeze(0).to(device)  # Shape: [sequence_length, input_size]
        song_labels = song_labels.squeeze(0).to(device)  # Shape: [sequence_length]

        # -- Initialize state units to 0 (will update w/ outputs in loop)
        state_units = torch.zeros((1, model.output_size)).to(device)

        # -- Iterate through each song timestep (each 1st/3rd beat)
        for timestep in range(song_inputs.size(0)):
            # Get melody input/chord label for current timestep
            input_t = song_inputs[timestep].unsqueeze(0)
            label_t = song_labels[timestep].unsqueeze(0)

            # Define meter_units based on batch index
            meter_units = F.one_hot(torch.arange(2, dtype=torch.long))[timestep % 2].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
            meter_units = meter_units.expand((dataloader.batch_size, 2))                        # Create batch dimension

            # Concatenate state_units, melody inputs, and meter_units
            inputs = torch.cat([state_units, input_t, meter_units], dim=1)

            # print(f"inputs: {inputs}")

            # Forward pass
            output = model(inputs)

            # Determine if chord prediction is correct
            # print(f"label_t: {label_t}, output: {output}")

            label_idx = label_t.item()
            chord_idx = torch.argmax(output.detach().squeeze(0)).item()

            # print(f"label_idx: {label_idx}, chord_idx: {chord_idx}")

            if chord_idx == label_idx:
                num_correct += 1

            # Update state units
            state_units = F.softmax(output, dim=1) + model.state_units_decay * state_units
            state_units = state_units / state_units.sum(dim=1, keepdim=True)

    # -- Compute testing accuracy
    total_samples = sum([len(song_inputs[0]) for song_inputs, _ in dataloader])
    accuracy = num_correct / total_samples

    return accuracy