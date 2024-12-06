import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from collections import deque

from models.hnn import HNN
from models.mnet import MelodyNet
from utils.data.load_data import get_song_key
from utils.data.mappings import *
from utils.data.distributions import get_label_feature

from tqdm import tqdm


def train(model: HNN | MelodyNet, dataloader: DataLoader, keys: list[str], criterion: nn.Module, optimizer: opt.Optimizer, device: torch.device):

    # -- Define meter units size based on model
    if isinstance(model, HNN):
        meter_size = 2
    elif isinstance(model, MelodyNet):
        meter_size = 16

    # -- Set model to train
    model.train()

    total_loss = 0.0

    # -- Iterate through DataLoader batches (each batch is a song)
    for song_key, (song_inputs, song_labels) in tqdm(zip(keys, dataloader), total=len(dataloader)):
        # -- Convert song key encoding to note string (e.g. 'A')
        # song_key_str = NOTE_ENC_TO_NOTE_STR_REF.get(song_key)

        # -- Put song data onto device and remove batch dim
        song_inputs = song_inputs.squeeze(0).to(device)  # Shape: [sequence_length, input_size]
        song_labels = song_labels.squeeze(0).to(device)  # Shape: [sequence_length]

        # -- Initialize state units to 0 (will update w/ outputs in loop)
        state_units = torch.zeros((1, model.output_size)).to(device)

        # -- Accumulate loss per song
        song_loss = 0.0

        # -- Accumulate repetition penalty from past 4 outputs
        past_outputs = deque(maxlen=4)

        # -- Track timestep to determine meter units
        timestep = 0

        for sample_idx in range(song_inputs.size(0)):
            # -- Get melody input/chord label for current timestep
            input_t = song_inputs[sample_idx].unsqueeze(0)
            label_t = song_labels[sample_idx].unsqueeze(0)

            # -- Define meter_units based on batch index
            meter_units = F.one_hot(torch.arange(meter_size, dtype=torch.long))[timestep % meter_size].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
            meter_units = meter_units.expand((dataloader.batch_size, meter_size))                        # Create batch dimension

            # -- Inject noise if True
            if isinstance(model, MelodyNet) and model.inject_noise:
                if model.inject_noise:
                    noise = torch.randn((1, model.noise_size)).to(device) * model.noise_weight
                    # Concatenate state_units, melody inputs, and meter_units
                    inputs = torch.cat([state_units, input_t, noise, meter_units], dim=1)
            else:
                # Concatenate state_units, melody inputs, and meter_units
                inputs = torch.cat([state_units, input_t, meter_units], dim=1)

            # -- Zero parameter gradients
            optimizer.zero_grad()

            # -- Forward pass
            output = model(inputs)

            # -- Compute loss
            loss = criterion(output, label_t)

            if isinstance(model, MelodyNet):
                # Determine predicted output class (as tensor)
                output_idx = torch.argmax(output, dim=1).item()

                # Add repetition penalty
                if model.repetition_loss != 0:
                    repetitions = sum([past_output == output_idx for past_output in past_outputs])

                    if len(past_outputs) > 0:
                        repetition_penalty = (repetitions / len(past_outputs)) * model.repetition_loss
                    else:
                        repetition_penalty = 0.0

                    past_outputs.append(output_idx)

                    loss += repetition_penalty

                # Add key penalty
                if model.key_loss != 0:
                    # Convert note and key to strings
                    output_note = get_label_feature(output_idx, 'notes')

                    song_key_str = get_song_key(song_key, song_labels)

                    if output_note not in KEY_NOTES.get(song_key_str) and output_note != 'R':
                        loss += model.key_loss

                # Add chord harmony penalty
                if model.harmony_loss != 0:
                    # Convert output note/chord to strings
                    output_note = get_label_feature(output_idx, 'notes')
                    
                    chord_idx_to_str = {v:k for k,v in CHORD_STR_TO_IDX_MNET.items()}
                    chord_idx = torch.argmax(input_t, dim=1).item()
                    chord = chord_idx_to_str[chord_idx]

                    if output_note not in HARMONIZING_NOTES.get(chord):
                        loss += model.harmony_loss


            # if isinstance(model, HNN):
            #     loss = criterion(output, label_t)
            # elif isinstance(model, MelodyNet) and model.repetition_weight != 0:

            #     repetitions = sum([(past_output == output_idx).sum().item() for past_output in past_outputs])

            #     if len(past_outputs) > 0:
            #         repetition_penalty = (repetitions / len(past_outputs)) * model.repetition_weight
            #     else:
            #         repetition_penalty = 0.0

            #     past_outputs.append(output_idx)

            #     loss = criterion(output, label_t) + repetition_penalty
            # else:
            #     loss = criterion(output, label_t)
                
                # loss = criterion(output, label_t)

                # Penalize incorrect predictions of rest notes
                # _, pred_idx = torch.max(output, dim=1)

                # if pred_idx.item() == 0 and label_t.item() != 0:
                #     loss *= model.rest_loss_weight

            # loss = criterion(output, label_t)

            song_loss += loss.item()

            # Backward pass
            loss.backward()

            # Zero fixed weight gradients
            if isinstance(model, HNN):
                model.hidden2_from_melody.weight.grad.fill_diagonal_(0.0)
            elif isinstance(model, MelodyNet) and model.fixed_chords:
                model.hidden2_from_chord.weight.grad.fill_diagonal_(0.0)                

            # Update weights
            optimizer.step()

            # Update state units w/ outputs softmax and normalize
            output_softmax = F.softmax(output.detach() / model.temperature, dim=1).to(device)

            state_units = output_softmax + model.state_units_decay * state_units
            state_units = state_units / state_units.sum(dim=1, keepdim=True)

            # Increment timestep by the duration of the label note
            note_idx_to_str = {v:k for k,v in NOTE_STR_TO_IDX_MNET.items()}
            note_str = note_idx_to_str[label_t.item()]

            if note_str[1] == '#':
                note_duration = int(note_str[3:])
            else:
                note_duration = int(note_str[2:])

            timestep += note_duration

        # Add batch loss to total loss
        total_loss += song_loss

    # -- Compute average epoch loss
    epoch_loss = total_loss / len(dataloader)

    return epoch_loss


def test(model: HNN | MelodyNet, dataloader: DataLoader, keys: list[str], device: torch.device):

    # -- Define meter units size based on model
    if isinstance(model, HNN):
        meter_size = 2
    elif isinstance(model, MelodyNet):
        meter_size = 16

    # -- Set model to eval mode
    model.eval()

    num_correct = 0
    num_correct_keys = 0
    num_correct_notes = 0

    print(f"keys len: {len(keys)}")
    print(f"dataloader len: {len(dataloader.dataset)}")

    # -- Iterate through testing DataLoader
    for song_key, (song_inputs, song_labels) in tqdm(zip(keys, dataloader), total=len(dataloader)):

        # -- Put song data onto device and add batch dim
        song_inputs = song_inputs.squeeze(0).to(device)  # Shape: [sequence_length, input_size]
        song_labels = song_labels.squeeze(0).to(device)  # Shape: [sequence_length]

        # -- Initialize state units to 0 (will update w/ outputs in loop)
        state_units = torch.zeros((1, model.output_size)).to(device)

        # -- Iterate through each song timestep
        timestep = 0

        while timestep < song_inputs.size(0):
            # Get melody input/chord label for current timestep
            input_t = song_inputs[timestep].unsqueeze(0)
            label_t = song_labels[timestep].unsqueeze(0)

            # Define meter_units based on timestep
            meter_units = F.one_hot(torch.arange(meter_size, dtype=torch.long))[timestep % meter_size].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
            meter_units = meter_units.expand((dataloader.batch_size, meter_size))                        # Create batch dimension

            # Inject noise if True
            if isinstance(model, MelodyNet) and model.inject_noise:
                noise = torch.randn((1, model.noise_size)).to(device) * model.noise_weight
                # Concatenate state_units, melody inputs, and meter_units
                inputs = torch.cat([state_units, input_t, noise, meter_units], dim=1)
            else:
                # Concatenate state_units, melody inputs, and meter_units
                inputs = torch.cat([state_units, input_t, meter_units], dim=1)

            # Forward pass
            output = model(inputs)

            label_idx = label_t.item()
            pred_idx = torch.argmax(output.detach().squeeze(0)).item()

            # -- Count number of correct label predictions
            if pred_idx == label_idx:
                num_correct += 1

            # -- Count number of predictions in the right key
            pred_note = get_label_feature(pred_idx, 'notes')
            label_note = get_label_feature(label_idx, 'notes')

            song_key_str = get_song_key(song_key, song_labels)

            if pred_note in KEY_NOTES.get(song_key_str) or pred_note == 'R':
                num_correct_keys += 1

            # -- Count number of predictions of the right note (even if octave is wrong)
            if pred_note == label_note:
                num_correct_notes += 1

            # Update state units
            state_units = F.softmax(output, dim=1) + model.state_units_decay * state_units
            state_units = state_units / state_units.sum(dim=1, keepdim=True)

            # Increment timestep by the duration of the label note
            note_idx_to_str = {v:k for k,v in NOTE_STR_TO_IDX_MNET.items()}
            note_str = note_idx_to_str[label_t.item()]

            if note_str[1] == '#':
                note_duration = int(note_str[3:])
            else:
                note_duration = int(note_str[2:])

            timestep += note_duration 

    # -- Compute testing accuracy
    total_samples = sum([len(song_inputs[0]) for song_inputs, _ in dataloader])

    accuracy = num_correct / total_samples
    key_accuracy = num_correct_keys / total_samples
    note_accuracy = num_correct_notes / total_samples

    return accuracy, key_accuracy, note_accuracy