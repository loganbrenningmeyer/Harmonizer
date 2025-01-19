import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as opt
from torch.utils.data import DataLoader
from collections import deque
import pandas as pd

import os

from models.hnn import HNN
from models.mnet import MelodyNet
from models.mnetRL import MelodyNetRL
from utils.data.load_data import get_song_key, parse_formatted_data, parse_data
from utils.data.mappings import *
from utils.data.distributions import get_label_feature

# -- RL Melody Quality Heuristics
from quality import Q, M_idx_to_enc, C_idx_to_enc

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

            # song_key_str = get_song_key(song_key, song_labels)

            # if pred_note in KEY_NOTES.get(song_key_str) or pred_note == 'R':
            #     num_correct_keys += 1

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



def eval_heuristics(model: MelodyNet | MelodyNetRL, train_dataloader_RL: DataLoader, test_dataloader_RL: DataLoader, heuristic_params: dict = None):

    if heuristic_params is None:
        heuristic_params = {
            'Q_h': {'P_chord': 0, 'P_scale': 2, 'P_diss': 8},
            'Q_s': {'w_Comp': 0.2, 'w_RR_I': 0.2, 'w_RR_D': 0.2, 'w_H_I': 0.2, 'w_H_D': 0.2, 'RR_I_thresh': 0.8, 'sigma': 3.0},
            'Q_b': {'w_chord': 0.34, 'w_scale': 0.33, 'w_diss': 0.33},
            'Q_f': {'P_unison': 5, 'P_stepwise': 1, 'P_conjunct': 2, 'P_disjunct': 3},
            'Q_M': {'w_h': 0.4, 'w_s': 0.2, 'w_b': 0.2, 'w_f': 0.2}
        }

    # -- Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Load MelodyNet model in eval mode
    model.to(device)
    model.eval()

    dataloaders = {
        'train': train_dataloader_RL,
        'test' : test_dataloader_RL
    }

    for split, dataloader in dataloaders.items():
        '''
        Track changes in heuristics over each iteration (song), to see which
        have most influence on training
        '''
        # -- Melody Quality
        Q_M_vals = []
        # -- Harmoniousness
        Q_h_vals = []
        num_chord_vals = []
        num_scale_vals = []
        num_diss_vals = []
        # -- Structure
        Q_s_vals = []
        Comp_M_vals = []
        RR_I_vals = []
        RR_D_vals = []
        H_I_norm_vals = []
        H_D_norm_vals = []
        # -- Balance
        Q_b_vals = []
        EMD_chord_vals = []
        EMD_scale_vals = []
        EMD_diss_vals = []
        # -- Flow
        Q_f_vals = []
        num_unison_vals = []
        num_stepwise_vals = []
        num_conjunct_vals = []
        num_disjunct_vals = []
        P_total_vals = []
        P_max_vals = []


        for song_inputs, song_labels in tqdm(dataloader):
            song_inputs = song_inputs.squeeze(0).to(device)

            if isinstance(model, MelodyNet):
                # -- Initialize state units to 0 (will update w/ outputs in loop)
                state_units = torch.zeros((1, model.output_size)).to(device)
            elif isinstance(model, MelodyNetRL):
                no_note_token = model.vocab_size                            # Index 459, outside of class indices [0, 458]
                past_notes = torch.full((model.n_past_notes,), 
                                        no_note_token,
                                        dtype=torch.long).to(device)        # Array of the n past class indices
                
                state_units = model.embedding_layer(past_notes).view(-1).unsqueeze(0)    # Shape: [n_past_notes * embedding_dim] = [8 * 32]

            # -- Initialize timestep to 0
            timestep = 0

            # -- Initialize melody and chord encoding arrays
            M = []
            C = []

            while timestep < len(song_inputs):
                input_t = song_inputs[timestep].unsqueeze(0)

                meter_units = F.one_hot(torch.arange(model.meter_size, dtype=torch.long))[timestep % model.meter_size].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
                meter_units = meter_units.expand((dataloader.batch_size, model.meter_size))

                # -- Concatenate state_units, melody inputs, and meter_units
                inputs = torch.cat([state_units, input_t, meter_units], dim=1)

                # -- Forward pass
                output = model(inputs)

                # Update state units w/ outputs softmax and normalize
                output_softmax = F.softmax(output.detach(), dim=1).to(device)

                # Append m_t and c_t
                m_t = torch.argmax(output.detach().squeeze(0)).item()
                M.append(m_t)

                c_t = torch.argmax(input_t.squeeze(0)).item()
                C.append(c_t)

                if isinstance(model, MelodyNet):
                    state_units = output_softmax + model.state_units_decay * state_units
                    state_units = state_units / state_units.sum(dim=1, keepdim=True)
                elif isinstance(model, MelodyNetRL):
                    past_notes = torch.cat([past_notes[1:], m_t.view(-1)], dim=0)  # Shift and append
                    state_units = model.embedding_layer(past_notes).view(-1).unsqueeze(0)  # Shape: [n_past_notes * embedding_dim]

                # Increment timestep by the duration of the predicted note
                duration = get_label_feature(m_t, 'durations')

                timestep += duration

            '''
            Compute melody quality heuristic Q(M) after song is complete
            '''
            M = M_idx_to_enc(M)
            C = C_idx_to_enc(C)

            '''
            Compute Heuristics and Record Submetrics for Each
            '''
            heuristics = Q(M, C, heuristic_params)

            # -- Melody Quality
            Q_M = heuristics['Q_M']
            # -- Harmoniousness
            Q_h = heuristics['Q_h'][0]
            num_chord, num_scale, num_diss = heuristics['Q_h'][1]
            # -- Structure
            Q_s = heuristics['Q_s'][0]
            Comp_M, RR_I, RR_D, H_I_norm, H_D_norm = heuristics['Q_s'][1]
            # -- Balance
            Q_b = heuristics['Q_b'][0]
            EMD_chord, EMD_scale, EMD_diss = heuristics['Q_b'][1]
            # -- Flow
            Q_f = heuristics['Q_f'][0]
            num_unison, num_stepwise, num_conjunct, num_disjunct, P_total, P_max = heuristics['Q_f'][1]

            # -- Record each heuristic's value
            Q_M_vals.append(Q_M)
            # -- Harmoniousness
            Q_h_vals.append(Q_h)
            num_chord_vals.append(num_chord)
            num_scale_vals.append(num_scale)
            num_diss_vals.append(num_diss)
            # -- Structure
            Q_s_vals.append(Q_s)
            Comp_M_vals.append(Comp_M)
            RR_I_vals.append(RR_I)
            RR_D_vals.append(RR_D)
            H_I_norm_vals.append(H_I_norm)
            H_D_norm_vals.append(H_D_norm)
            # -- Balance
            Q_b_vals.append(Q_b)
            EMD_chord_vals.append(EMD_chord)
            EMD_scale_vals.append(EMD_scale)
            EMD_diss_vals.append(EMD_diss)
            # -- Flow
            Q_f_vals.append(Q_f)
            num_unison_vals.append(num_unison)
            num_stepwise_vals.append(num_stepwise)
            num_conjunct_vals.append(num_conjunct)
            num_disjunct_vals.append(num_disjunct)
            P_total_vals.append(P_total)
            P_max_vals.append(P_max)

        
        '''
        Write Heuristics Training Values to csv
        '''
        heuristics = {
            # -- Melody Quality
            'Q_M': Q_M_vals,
            'Q_h': Q_h_vals,
            'Q_s': Q_s_vals,
            'Q_b': Q_b_vals,
            'Q_f': Q_f_vals,
            # -- Harmoniousness Components
            'num_chord': num_chord_vals,
            'num_scale': num_scale_vals,
            'num_diss': num_diss_vals,
            # -- Structure Components
            'Comp_M': Comp_M_vals,
            'RR_I': RR_I_vals,
            'RR_D': RR_D_vals,
            'H_I_norm': H_I_norm_vals,
            'H_D_norm': H_D_norm_vals,
            # -- Balance Compone# nts
            'EMD_chord': EMD_chord_vals,
            'EMD_scale': EMD_scale_vals,
            'EMD_diss': EMD_diss_vals,
            # -- Flow Components
            'num_unison': num_unison_vals,
            'num_stepwise': num_stepwise_vals,
            'num_conjunct': num_conjunct_vals,
            'num_disjunct': num_disjunct_vals,
            'P_total': P_total_vals,
            'P_max': P_max_vals
        }

        df = pd.DataFrame(heuristics)
        if isinstance(model, MelodyNet):
            save_path = f'saved_models/mnet/{model.model_name}/eval'
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(f'{save_path}/{split}_heuristics_eval.csv')
        elif isinstance(model, MelodyNetRL):
            save_path = f'saved_models/mnetRL/{model.model_name}/eval'
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(f'{save_path}/{split}_heuristics_eval.csv')




'''
----- MelodyNetRL Training -----
'''
def train_mnetRL(model: MelodyNetRL, 
                 SL_dataloader: DataLoader, 
                 RL_dataloader: DataLoader,
                 criterion: nn.Module, 
                 optimizer: opt.Optimizer, 
                 device: torch.device):

    '''
    Phase 1: Supervised Learning

    - State Units and Timestep updated based on [Ground Truth]
    - Loss/Updates: Applied on a note-level basis from [Ground Truth]
    '''
    print("Phase 1: Supervised Learning...\n")

    # -- Set model to train
    model.train()

    SL_epoch_loss = 0.0

    for song_inputs, song_labels in tqdm(SL_dataloader):
        # -- Track song loss
        SL_song_loss = 0.0

        # -- Put song data onto device and remove batch dim
        song_inputs = song_inputs.squeeze(0).to(device)  # Shape: [sequence_length, input_size]
        song_labels = song_labels.squeeze(0).to(device)  # Shape: [sequence_length]

        '''
        Initialize State Units with 'no-note' token to represent an empty
        note context

        SL: State Units and Timestep updated based on [Ground Truth]
        '''
        # state_units = torch.zeros((1, model.output_size)).to(device)
        no_note_token = model.vocab_size                            # Index 459, outside of class indices [0, 458]
        past_notes = torch.full((model.n_past_notes,), 
                                no_note_token,
                                dtype=torch.long).to(device)        # Array of the n past class indices
        
        state_units = model.embedding_layer(past_notes).view(-1).unsqueeze(0)    # Shape: [n_past_notes * embedding_dim] = [8 * 32]

        # -- Initialize timestep
        timestep = 0

        # -- Iterate over song samples
        for sample_idx in range(song_inputs.size(0)):
            # -- Get melody input/chord label for current timestep
            input_t = song_inputs[sample_idx].unsqueeze(0)
            label_t = song_labels[sample_idx].unsqueeze(0)

            # -- Determine meter index based on timestep (SL based on [Ground Truth])
            meter_idx = timestep % 16
            meter_units = F.one_hot(torch.tensor(meter_idx, dtype=torch.long), num_classes=16).to(device)
            meter_units = meter_units.float().unsqueeze(0)

            # -- Concatenate state_units, melody inputs, and meter_units
            inputs = torch.cat([state_units, input_t, meter_units], dim=1)

            # -- Forward pass
            output = model(inputs)

            # -- Compute SL Loss (Cross Entropy)
            loss = criterion(output, label_t)
            SL_song_loss += loss.item()

            # -- Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # -- Zero fixed weight gradients
            model.hidden2_from_chord.weight.grad.fill_diagonal_(0.0)

            # -- Update parameters
            optimizer.step()

            '''
            SL: Timestep updated based on [Ground Truth] label_t
            '''
            duration = get_label_feature(label_t.item(), 'durations')
            timestep += duration

            '''
            SL: Update State Units based on [Ground Truth] label_t
            '''
            # state_units = F.softmax(output.detach(), dim=1).to(device) + model.state_units_decay * state_units
            # state_units = state_units / state_units.sum(dim=1, keepdim=True)
            past_notes = torch.cat([past_notes[1:], label_t.view(-1)], dim=0) # Shift label_t into past_notes
            state_units = model.embedding_layer(past_notes).view(-1).unsqueeze(0)    # Embed past n notes for state units [n_past_notes * embedding_dim]

        # print(f"SL Song Loss: {SL_song_loss}")
        SL_epoch_loss += SL_song_loss

    print(f"SL Epoch Loss: {SL_epoch_loss}, SL Song Average: {SL_epoch_loss / len(SL_dataloader)}")

    # -- Save supervised model
    weights_dir = f'saved_models/mnetRL/SL_{model.model_name}/weights'
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(model, f'{weights_dir}/epoch1.pth')
    print(f"Saved SL_{model.model_name} epoch1.pth to {weights_dir}/epoch1.pth.\n")
    '''
    Phase 2: Reinforcement Learning

    - State Units and Timestep updated based on [Model Actions]
    - Loss/Updates: Applied on a song-level basis based on Melody Quality heuristics
    '''
    print("Phase 2: Reinforcement Learning...\n")

    # -- Set model to train
    model.train()

    RL_epoch_reward = 0.0

    '''
    Track changes in heuristics over each iteration (song), to see which
    have most influence on training
    '''
    # -- Melody Quality
    Q_M_vals = []
    # -- Harmoniousness
    Q_h_vals = []
    num_chord_vals = []
    num_scale_vals = []
    num_diss_vals = []
    # -- Structure
    Q_s_vals = []
    Comp_M_vals = []
    RR_I_vals = []
    RR_D_vals = []
    H_I_norm_vals = []
    H_D_norm_vals = []
    # -- Balance
    Q_b_vals = []
    EMD_chord_vals = []
    EMD_scale_vals = []
    EMD_diss_vals = []
    # -- Flow
    Q_f_vals = []
    num_unison_vals = []
    num_stepwise_vals = []
    num_conjunct_vals = []
    num_disjunct_vals = []
    P_total_vals = []
    P_max_vals = []

    for song_inputs, song_labels in tqdm(RL_dataloader):

        # -- Put song data onto device and remove batch dim
        song_inputs = song_inputs.squeeze(0).to(device)  # Shape: [sequence_length, input_size]
        # song_labels = song_labels.squeeze(0).to(device)  # Shape: [sequence_length]

        '''
        Initialize State Units with 'no-note' token to represent an empty
        note context

        RL: State Units and Timestep updated based on [Model Actions]
        '''
        # state_units = torch.zeros((1, model.output_size)).to(device)
        no_note_token = model.vocab_size                            # Index 459, outside of class indices [0, 458]
        past_notes = torch.full((model.n_past_notes,), 
                                no_note_token,
                                dtype=torch.long).to(device)        # Array of the n past class indices
        
        state_units = model.embedding_layer(past_notes).view(-1).unsqueeze(0)    # Shape: [n_past_notes * embedding_dim] = [8 * 32]

        # -- Initialize timestep
        timestep = 0

        '''
        RL: Updates applied on a song-level basis; log probs/actions should be stored
        over the full song
        '''
        log_probs = []
        actions = []
        M = []
        C = []

        # -- Progress through timesteps until song is complete
        total_timesteps = len(song_inputs)

        while(timestep < total_timesteps):
            # -- Get chord input for the current timestep
            input_t = song_inputs[timestep].unsqueeze(0)

            # -- Determine meter index based on timestep (RL based on [Model Action])
            meter_idx = timestep % model.meter_size
            meter_units = F.one_hot(torch.tensor(meter_idx, dtype=torch.long), num_classes=model.meter_size).to(device)
            meter_units = meter_units.float().unsqueeze(0)

            # -- Concatenate state_units, melody inputs, and meter_units
            inputs = torch.cat([state_units, input_t, meter_units], dim=1)

            # -- Forward pass
            output = model(inputs)

            # -- Apply softmax to get output probs
            probs = F.softmax(output, dim=-1)

            # -- Sample a Model Action from distribution
            dist = Categorical(probs)
            a_t = dist.sample()

            # -- Get log prob of the sampled action
            log_prob = dist.log_prob(a_t)
            log_probs.append(log_prob)

            # -- Store the sampled action 
            actions.append(a_t)

            # -- Store the action's corresponding melody note index (building melody M)
            m_t = a_t.item()
            M.append(m_t)

            # -- Store the timestep's current chord input encoding (building chords C)
            # chord_idx_to_str = {v:k for k,v in CHORD_STR_TO_IDX_MNET.items()}
            c_t = torch.argmax(input_t.squeeze(0)).item()
            # c_t = chord_idx_to_str[c_idx]
            C.append(c_t)
            
            '''
            RL: State Units updated based on [Model Action] a_t
            '''
            # state_units = F.softmax(output.detach(), dim=1).to(device) + model.state_units_decay * state_units
            # state_units = state_units / state_units.sum(dim=1, keepdim=True)
            past_notes = torch.cat([past_notes[1:], a_t.view(-1)], dim=0)  # Shift and append
            state_units = model.embedding_layer(past_notes).view(-1).unsqueeze(0)  # Shape: [n_past_notes * embedding_dim]

            # -- Obtain duration from sampled action
            duration = get_label_feature(a_t.item(), 'durations')

            '''
            RL: Timestep updated based on [Model Action] duration
            '''
            timestep += duration

        '''
        Compute melody quality heuristic Q(M) after song is complete
        '''
        # -- Convert M and C to encodings
        M = M_idx_to_enc(M)
        C = C_idx_to_enc(C)

        '''
        Compute Heuristics and Record Submetrics for Each
        '''
        heuristics = Q(M, C, model.heuristic_params)

        # -- Melody Quality
        Q_M = heuristics['Q_M']
        # -- Harmoniousness
        Q_h = heuristics['Q_h'][0]
        num_chord, num_scale, num_diss = heuristics['Q_h'][1]
        # -- Structure
        Q_s = heuristics['Q_s'][0]
        Comp_M, RR_I, RR_D, H_I_norm, H_D_norm = heuristics['Q_s'][1]
        # -- Balance
        Q_b = heuristics['Q_b'][0]
        EMD_chord, EMD_scale, EMD_diss = heuristics['Q_b'][1]
        # -- Flow
        Q_f = heuristics['Q_f'][0]
        num_unison, num_stepwise, num_conjunct, num_disjunct, P_total, P_max = heuristics['Q_f'][1]

        # -- Record each heuristic's value
        Q_M_vals.append(Q_M)
        # -- Harmoniousness
        Q_h_vals.append(Q_h)
        num_chord_vals.append(num_chord)
        num_scale_vals.append(num_scale)
        num_diss_vals.append(num_diss)
        # -- Structure
        Q_s_vals.append(Q_s)
        Comp_M_vals.append(Comp_M)
        RR_I_vals.append(RR_I)
        RR_D_vals.append(RR_D)
        H_I_norm_vals.append(H_I_norm)
        H_D_norm_vals.append(H_D_norm)
        # -- Balance
        Q_b_vals.append(Q_b)
        EMD_chord_vals.append(EMD_chord)
        EMD_scale_vals.append(EMD_scale)
        EMD_diss_vals.append(EMD_diss)
        # -- Flow
        Q_f_vals.append(Q_f)
        num_unison_vals.append(num_unison)
        num_stepwise_vals.append(num_stepwise)
        num_conjunct_vals.append(num_conjunct)
        num_disjunct_vals.append(num_disjunct)
        P_total_vals.append(P_total)
        P_max_vals.append(P_max)

        # -- Scale reward by reward_weight
        Q_M *= model.reward_weight

        '''
        Compute REINFORCE loss
        '''
        log_probs_tensor = torch.stack(log_probs)
        reinforce_loss = -log_probs_tensor.sum() * Q_M

        # -- Backward pass and optimization
        optimizer.zero_grad()
        reinforce_loss.backward()

        # -- Zero fixed weight gradients
        if model.hidden2_from_chord.weight.grad is not None:
                model.hidden2_from_chord.weight.grad.fill_diagonal_(0.0)

        # -- Update parameters
        optimizer.step()

        # print(f"RL Song Reward: {RL_song_reward}")
        RL_epoch_reward += Q_M

    print(f"RL Epoch Reward: {RL_epoch_reward}, RL Song Average: {RL_epoch_reward / len(RL_dataloader)}")


    '''
    Write Heuristics Training Values to csv
    '''
    heuristics = {
        # -- Melody Quality
        'Q_M': Q_M_vals,
        'Q_h': Q_h_vals,
        'Q_s': Q_s_vals,
        'Q_b': Q_b_vals,
        'Q_f': Q_f_vals,
        # -- Harmoniousness Components
        'num_chord': num_chord_vals,
        'num_scale': num_scale_vals,
        'num_diss': num_diss_vals,
        # -- Structure Components
        'Comp_M': Comp_M_vals,
        'RR_I': RR_I_vals,
        'RR_D': RR_D_vals,
        'H_I_norm': H_I_norm_vals,
        'H_D_norm': H_D_norm_vals,
        # -- Balance Compone# nts
        'EMD_chord': EMD_chord_vals,
        'EMD_scale': EMD_scale_vals,
        'EMD_diss': EMD_diss_vals,
        # -- Flow Components
        'num_unison': num_unison_vals,
        'num_stepwise': num_stepwise_vals,
        'num_conjunct': num_conjunct_vals,
        'num_disjunct': num_disjunct_vals,
        'P_total': P_total_vals,
        'P_max': P_max_vals
    }

    df = pd.DataFrame(heuristics)
    df.to_csv(f'saved_models/mnetRL/{model.model_name}/train_heuristics.csv')


    return SL_epoch_loss, RL_epoch_reward


def test_mnetRL():
    return
