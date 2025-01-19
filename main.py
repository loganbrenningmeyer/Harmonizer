import torch
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import re
from tqdm import tqdm


from models.hnn import HNN
from models.mnet import MelodyNet
from models.mnetRL import MelodyNetRL

from utils.data.load_data import *
from utils.data.distributions import *
from utils.compare_models import compare_plots

from utils.experiment import train, test, train_mnetRL, test_mnetRL, eval_heuristics
from quality import M_idx_to_enc, C_idx_to_enc, plot_model_comparison

from collections import Counter

'''
Model Parameters:

# -- hidden1_64_melody_10
hnn = HNN(hidden1_size=64, lr=0.05, weight_decay=1e-4,
            melody_weights=10.0, chord_weights=2.5, state_units_decay=0.75,
            model_name='hidden1_64_melody_10')

# -- hidden1_64_melody_10_state_05
hnn = HNN(hidden1_size=64, lr=0.05, weight_decay=1e-4,
            melody_weights=10.0, chord_weights=2.5, state_units_decay=0.5,
            model_name='hidden1_64_melody_10_state_05')

# -- hidden1_128
    hnn = HNN(hidden1_size=128, lr=0.05, weight_decay=1e-5,
              melody_weights=10.0, chord_weights=2.5, state_units_decay=0.75,
              model_name='hidden1_128')

# -- hidden1_128_state_05
    hnn = HNN(hidden1_size=128, lr=0.05, weight_decay=1e-5,
              melody_weights=10.0, chord_weights=2.5, state_units_decay=0.5,
              model_name='hidden1_128_state_05')

# -- hidden1_128_state_075
    hnn = HNN(hidden1_size=128, lr=0.05, weight_decay=1e-5,
              melody_weights=10.0, chord_weights=2.5, state_units_decay=0.75,
              model_name='hidden1_128_state_075')
'''


def hnn_main():
    '''
    Create DataLoaders
    '''
    train_dataloader, test_dataloader = create_dataloaders_hnn()

    '''
    Create HNN Model
    '''
    # -- Initialize model
    hnn = HNN(hidden1_size=256, lr=0.01, weight_decay=0.0,
              melody_weights=10.0, chord_weights=2.5, state_units_decay=0.75,
              model_name='huge_size_high_weights_low_lr')

    # -- Put model on device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hnn.to(device)

    # -- Define criterion/optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.SGD(hnn.parameters(), lr=hnn.lr, momentum=hnn.momentum, weight_decay=hnn.weight_decay)

    '''
    Training/Testing
    '''
    # -- Create model weights dir/model figs dir
    weights_dir = f'saved_models/hnn/{hnn.model_name}/weights'
    figs_dir = f'saved_models/hnn/{hnn.model_name}/figs'
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    epochs = 1000
    for epoch in range(1, epochs + 1):
        # -- Train for an epoch and store epoch loss
        epoch_loss = train(hnn, train_dataloader, criterion, optimizer, device)
        train_losses.append(epoch_loss)
        print(f"-- Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

        # -- Evaluate model on train set
        train_accuracy = test(hnn, train_dataloader, device)
        train_accuracies.append(train_accuracy * 100)
        print(f"Training Accuracy: {train_accuracy*100:.2f}%")

        # -- Evaluate model on test set
        test_accuracy = test(hnn, test_dataloader, device)
        test_accuracies.append(test_accuracy * 100)
        print(f"Testing Accuracy: {test_accuracy*100:.2f}%\n")

        # -- Save trained model every 10th epoch
        if epoch % 10 == 0:
            torch.save(hnn, f'{weights_dir}/epoch{epoch}.pth')

    '''
    Plot Training & Testing Loss/Accuracy
    '''
    # -- Plot training loss
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='red')

    plt.title(f'Train Loss ({hnn.model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{figs_dir}/train_loss.png')
    plt.close()

    # -- Plot training accuracy
    plt.figure()
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', color='green')

    plt.title(f'Train Accuracy ({hnn.model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{figs_dir}/train_accuracy.png')
    plt.close()

    # -- Plot testing accuracy
    plt.figure()
    plt.plot(range(1, epochs + 1), test_accuracies, label='Testing Accuracy', color='blue')

    plt.title(f'Test Accuracy ({hnn.model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{figs_dir}/test_accuracy.png')
    plt.close()

    # -- Coplot loss & accuracies
    plt.figure()

    # Plot training loss
    ax1 = plt.gca()
    ax1.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # Plot training/testing accuracy w/ shared x-axis
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', color='green')
    ax2.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='blue')
    ax2.set_ylabel('Accuracy (%)')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title(f'Train Loss & Train/Test Accuracy ({hnn.model_name})')
    plt.tight_layout()
    plt.savefig(f'{figs_dir}/train_test_loss_accuracy.png')
    plt.close()

    '''
    Write Training Metrics/Model Params to CSV
    '''
    metrics = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'test_accuracy': test_accuracies
    }

    params = {
        'hidden1_size': hnn.hidden1_size,
        'lr': hnn.lr,
        'weight_decay': hnn.weight_decay,
        'melody_weights': hnn.melody_weights,
        'chord_weights': hnn.chord_weights,
        'state_units_decay': hnn.state_units_decay
    }

    df = pd.DataFrame(metrics)
    df.to_csv(f'saved_models/hnn/{hnn.model_name}/metrics.csv', index=False)

    df = pd.DataFrame(params, index=[0])
    df.to_csv(f'saved_models/hnn/{hnn.model_name}/params.csv', index=False)

# weights = torch.ones(289).to(device)
    # weights[0] = mnet.rest_loss_weight

    # -- Initialize model
    # mnet = MelodyNet(hidden1_size=256, lr=0.05, weight_decay=0.0, repetition_weight=500.0,
    #                  chord_weight=20.0, melody_weight=5.0, state_units_decay=0.1, 
    #                  rest_fixed_weight=0.1, rest_loss_weight=1.0,
    #                  inject_noise=True, noise_size=100, noise_weight=1.0,
    #                  temperature=1.0,
    #                  model_name='noise_2')

def mnet_main():

    '''
    Create DataLoaders

    Available balance_feature options:
    - 'classes'
    - 'notes'
    - 'octaves'
    - 'notes_octaves'
    - 'durations'
    '''
    balance_feature = 'notes_octaves'

    train_dataloader, test_dataloader, train_song_keys, test_song_keys = create_dataloaders_mnet(balance_feature=balance_feature,
                                                                                             subset=1)

    '''
    Create MelodyNet Model
    '''
    mnet = MelodyNet(hidden1_size=512, lr=0.001, weight_decay=0.0, 
                     repetition_loss=500.0, key_loss=0.0, harmony_loss=0.0,
                     chord_weight=5.0, melody_weight=5.0, state_units_decay=0.5, 
                     fixed_chords=True, fixed_melody=True,
                     rest_fixed_weight=0.2, rest_loss_weight=1.0,
                     inject_noise=False, noise_size=100, noise_weight=1.0,
                     temperature=10.0, dropout_rate=0.5,
                     model_name='med_weights_comp')

    # -- Put model on device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mnet.to(device)

    # -- Define criterion/optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.SGD(mnet.parameters(), lr=mnet.lr, momentum=mnet.momentum, weight_decay=mnet.weight_decay)
    
    '''
    Store Params/Create Save Directories
    '''
    # -- Create model weights dir/model figs dir
    weights_dir = f'saved_models/mnet/{mnet.model_name}/weights'
    figs_dir = f'saved_models/mnet/{mnet.model_name}/figs'
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    
    params = {
        'hidden1_size': mnet.hidden1_size,
        'lr': mnet.lr,
        'weight_decay': mnet.weight_decay,
        'state_units_decay': mnet.state_units_decay,

        'repetition_loss': mnet.repetition_loss,
        'key_loss': mnet.key_loss,
        'harmony_loss': mnet.harmony_loss,
        'rest_fixed_weight': mnet.rest_fixed_weight,
        'rest_loss_weight': mnet.rest_loss_weight,

        'chord_weight': mnet.chord_weight,
        'melody_weight': mnet.melody_weight,
        'fixed_chords': mnet.fixed_chords,
        'fixed_melody': mnet.fixed_melody,

        'inject_noise': mnet.inject_noise,
        'noise_size': mnet.noise_size,
        'noise_weight': mnet.noise_weight,
        'temperature': mnet.temperature,
        'dropout_rate': mnet.dropout_rate,
        'balance_feature': balance_feature
    }

    df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
    df.to_csv(f'saved_models/mnet/{mnet.model_name}/params.csv', index=False, header=False)
    print(f"Saved {mnet.model_name} params.csv to saved_models/mnet/{mnet.model_name}/params.csv.")
    
    '''
    Training/Testing
    '''
    train_losses = []
    train_accuracies = []
    train_key_accuracies = []
    train_note_accuracies = []
    test_accuracies = []
    test_key_accuracies = []
    test_note_accuracies = []

    print(f"\n-- Training {mnet.model_name}...\n")

    epochs = 1
    for epoch in range(1, epochs + 1):
        # -- Train for an epoch and store epoch loss
        epoch_loss = train(mnet, train_dataloader, train_song_keys, criterion, optimizer, device)
        train_losses.append(epoch_loss)
        print(f"-- Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

        # -- Save trained model every epoch
        torch.save(mnet, f'{weights_dir}/epoch{epoch}.pth')
        print(f"Saved {mnet.model_name} epoch{epoch}.pth to {weights_dir}/epoch{epoch}.pth.")

        # -- Evaluate model on train set
        train_accuracy, train_key_accuracy, train_note_accuracy = test(mnet, train_dataloader, train_song_keys, device)
        train_accuracies.append(train_accuracy * 100)
        train_key_accuracies.append(train_key_accuracy * 100)
        train_note_accuracies.append(train_note_accuracy * 100)
        print(f"Training Accuracy: {train_accuracy*100:.2f}%, Training Key Accuracy: {train_key_accuracy*100:.2f}%, Training Note Accuracy: {train_note_accuracy*100:.2f}%")

        # -- Evaluate model on test set
        test_accuracy, test_key_accuracy, test_note_accuracy = test(mnet, test_dataloader, test_song_keys, device)
        test_accuracies.append(test_accuracy * 100)
        test_key_accuracies.append(test_key_accuracy * 100)
        test_note_accuracies.append(test_note_accuracy * 100)
        print(f"Testing Accuracy: {test_accuracy*100:.2f}%, Testing Key Accuracy: {test_key_accuracy*100:.2f}%, Testing Note Accuracy: {test_note_accuracy*100:.2f}%\n")

        metrics = {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'train_key_accuracy': train_key_accuracies,
            'train_note_accuracy': train_note_accuracies,
            'test_accuracy': test_accuracies,
            'test_key_accuracy': test_key_accuracies,
            'test_note_accuracy': test_note_accuracies
        }

        df = pd.DataFrame(metrics)
        df.to_csv(f'saved_models/mnet/{mnet.model_name}/metrics.csv', index=False)
        print(f"Saved {mnet.model_name} metrics.csv to saved_models/mnet/{mnet.model_name}/metrics.csv.") 


    '''
    Plot Training & Testing Loss/Accuracy
    '''
    # -- Plot training loss
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='red')

    plt.title(f'Train Loss ({mnet.model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{figs_dir}/train_loss.png')
    plt.close()

    # -- Plot training accuracy
    plt.figure()
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', color='green')

    plt.title(f'Train Accuracy ({mnet.model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{figs_dir}/train_accuracy.png')
    plt.close()

    # -- Plot testing accuracy
    plt.figure()
    plt.plot(range(1, epochs + 1), test_accuracies, label='Testing Accuracy', color='blue')

    plt.title(f'Test Accuracy ({mnet.model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{figs_dir}/test_accuracy.png')
    plt.close()

    # -- Coplot loss & accuracies
    plt.figure()

    # Plot training loss
    ax1 = plt.gca()
    ax1.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # Plot training/testing accuracy w/ shared x-axis
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', color='green')
    ax2.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='blue')
    ax2.set_ylabel('Accuracy (%)')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title(f'Train Loss & Train/Test Accuracy ({mnet.model_name})')
    plt.tight_layout()
    plt.savefig(f'{figs_dir}/train_test_loss_accuracy.png')
    plt.close()



def mnetRL_main():
    balance_feature = 'notes_octaves'

    SL_train_dataloader, SL_test_dataloader, _, _ = create_dataloaders_mnet(subset=1, balance_feature=balance_feature)     # Note-by-note
    RL_train_dataloader, RL_test_dataloader = create_dataloaders_mnetRL(subset=1)   # All 16th-beat timesteps

    '''
    Create MelodyNetRL model
    '''
    heuristic_params = {
        'Q_h': {'P_chord': 0, 'P_scale': 2, 'P_diss': 8},
        'Q_s': {'w_Comp': 0.2, 'w_RR_I': 0.2, 'w_RR_D': 0.2, 'w_H_I': 0.2, 'w_H_D': 0.2, 'RR_I_thresh': 0.8, 'sigma': 3.0},
        'Q_b': {'w_chord': 0.34, 'w_scale': 0.33, 'w_diss': 0.33},
        'Q_f': {'P_unison': 10, 'P_stepwise': 1, 'P_conjunct': 2, 'P_disjunct': 8},
        'Q_M': {'w_h': 0.4, 'w_s': 0.2, 'w_b': 0.2, 'w_f': 0.2}
    }

    mnetRL = MelodyNetRL(hidden1_size=1024, lr=0.0005, weight_decay=0.0,
                         chord_weight=10.0, melody_weight=15.0, rest_fixed_weight=0.25,
                         fixed_chords=True, fixed_melody=True, dropout_rate=0.0, 
                         reward_weight=10, heuristic_params=heuristic_params,
                         model_name="adam_high_flow")
    
    # -- Put model on device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mnetRL.to(device)

    # -- Define criterion/optimizer
    criterion = nn.CrossEntropyLoss()       # For Supervised Stage Only
    optimizer = torch.optim.Adam(mnetRL.parameters(), lr=0.001, weight_decay=mnetRL.weight_decay)
    # optimizer = opt.SGD(mnetRL.parameters(), 
    #                     lr=mnetRL.lr, 
    #                     momentum=0.0, 
    #                     weight_decay=mnetRL.weight_decay)
    
    '''
    Store Params/Create Save Directories
    '''
    # -- Create model weights dir/model figs dir
    weights_dir = f'saved_models/mnetRL/{mnetRL.model_name}/weights'
    figs_dir = f'saved_models/mnetRL/{mnetRL.model_name}/figs'
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    params = {
        'hidden1_size': mnetRL.hidden1_size,
        'lr': mnetRL.lr,
        'weight_decay': mnetRL.weight_decay,
        'chord_weight': mnetRL.chord_weight,
        'melody_weight': mnetRL.melody_weight,
        'rest_fixed_weight': mnetRL.rest_fixed_weight,
        'fixed_chords': mnetRL.fixed_chords,
        'fixed_melody': mnetRL.fixed_melody,
        'dropout_rate': mnetRL.dropout_rate,
        'balance_feature': balance_feature
    }

    # -- Write heuristic parameters to csv
    heuristic_params_flat = {f"{key1}_{key2}": value for key1, value1 in heuristic_params.items() for key2, value in value1.items()}
    params.update(heuristic_params_flat)

    df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
    df.to_csv(f'saved_models/mnetRL/{mnetRL.model_name}/params.csv', index=False, header=True)
    print(f"Saved {mnetRL.model_name} params.csv to saved_models/mnetRL/{mnetRL.model_name}/params.csv.")

    '''
    Training/Testing
    '''
    print(f"\n-- Training {mnetRL.model_name}...\n")

    epochs = 1
    for epoch in range(1, epochs + 1):
        SL_epoch_loss, RL_epoch_reward = train_mnetRL(model=mnetRL,
                                                      SL_dataloader=SL_train_dataloader,
                                                      RL_dataloader=RL_train_dataloader,
                                                      criterion=criterion,
                                                      optimizer=optimizer,
                                                      device=device)
        print(f"-- Epoch {epoch}/{epochs}, Loss: {SL_epoch_loss:.4f}, Reward: {RL_epoch_reward:.4f}")

        # -- Save trained model every epoch
        torch.save(mnetRL, f'{weights_dir}/epoch{epoch}.pth')
        print(f"Saved {mnetRL.model_name} epoch{epoch}.pth to {weights_dir}/epoch{epoch}.pth.")





# pattern = re.compile(r'^(R|[A-G]#?)(\d+)(\d+)$')

# note_order = ['R', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

# note_order_map = {note: index for index, note in enumerate(note_order)}

# def sort_key(item):
#     match = pattern.match(item)
#     if not match:
#         # Handle unexpected format by placing it at the end
#         return (len(note_order), 0, 0)
    
#     note, octave, duration = match.groups()
    
#     # Get the order index of the note; if not found, place it at the end
#     note_idx = note_order_map.get(note, len(note_order))
    
#     # Convert octave and duration to integers for proper sorting
#     octave = int(octave)
#     duration = int(duration)
    
#     return (note_idx, octave, duration)


if __name__ == "__main__":

    model_paths = ['saved_models/mnet/notes_octaves_balance_high_mel_rep_low_lr',
                   'saved_models/mnet/both_chords_melody_low_lr',
                   'saved_models/mnet/why_not_take2_full']
    
    # for model_path in model_paths:
    #     model = torch.load(os.path.join(model_path, 'weights/epoch1.pth'))
    #     train_dataloader, test_dataloader = create_dataloaders_mnetRL(shuffle=False)
    #     eval_heuristics(model, train_dataloader, test_dataloader)
        
    plot_model_comparison(model_paths, output_dir='figs/plot_test')
    # # mnetRL_main()
    # heuristic_params = {
    #     'Q_h': {'P_chord': 0, 'P_scale': 2, 'P_diss': 8},
    #     'Q_s': {'w_Comp': 0.2, 'w_RR_I': 0.2, 'w_RR_D': 0.2, 'w_H_I': 0.2, 'w_H_D': 0.2, 'RR_I_thresh': 0.8, 'sigma': 3.0},
    #     'Q_b': {'w_chord': 0.34, 'w_scale': 0.33, 'w_diss': 0.33},
    #     'Q_f': {'P_unison': 5, 'P_stepwise': 1, 'P_conjunct': 2, 'P_disjunct': 3},
    #     'Q_M': {'w_h': 0.4, 'w_s': 0.2, 'w_b': 0.2, 'w_f': 0.2}
    # }
   
    # # mnet_main()
    # mnet = torch.load('saved_models/mnet/low_weights_comp/weights/epoch1.pth')
    # RL_train_dataloader, RL_test_dataloader = create_dataloaders_mnetRL(subset=1, shuffle=False)   # All 16th-beat timesteps

    # eval_heuristics_mnet(mnet, RL_train_dataloader, heuristic_params)
    
    # M_indices = [50, 1, 2]
    # C_indices = [0, 1, 2]
    # M = M_idx_to_enc(M_indices)
    # C = C_idx_to_enc(C_indices)

    # print(M)
    # print(C)

    # num_chord_classes = 84
    # chord_one_hot = torch.from_numpy(np.eye(num_chord_classes, dtype=int))[0]

    # print(torch.argmax(chord_one_hot).item())
    # parse_data(hnn_data=False)

    # compare_plots()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_dataloader, test_dataloader, test_song_keys = create_dataloaders_mnet()

    # print(f"song_keys len: {len(test_song_keys)}")
    # print(f"dataloader len: {len(test_dataloader.dataset)}")

    # for song_idx, (song_inputs, song_labels) in tqdm(enumerate(test_dataloader)):
    #     song_key = NOTE_ENC_TO_NOTE_STR_REF.get(test_song_keys[song_idx])


    # model_names = [
    #     'no_fixed_chords_low_lr',
    #     'no_fixed_melody_low_lr',
    #     'no_fixed_chords_melody_low_lr',
    #     'both_chords_melody_low_lr',
    # ]
    
    # for model_name in model_names:

    #     print(f"Testing {model_name}...")

    #     mnet = torch.load(f'saved_models/mnet/{model_name}/weights/epoch1.pth')

    #     test_accuracy, test_key_accuracy, test_note_accuracy = test(mnet, test_dataloader, test_song_keys, device)

    #     print(f"Testing Accuracy: {test_accuracy*100:.2f}%, Testing Key Accuracy: {test_key_accuracy*100:.2f}%, Testing Note Accuracy: {test_note_accuracy*100:.2f}%\n")

    '''
    plot_counts()
    '''
    # train_dataloader, test_dataloader, train_song_keys, test_song_keys = create_dataloaders_mnet()

    # label_features = ['classes', 'notes', 'octaves', 'notes_octaves', 'durations']

    # for label_feature in label_features:

    #     orig_save_dir = f'figs/distributions/mnet/{label_feature}/original'
    #     new_save_dir = f'figs/distributions/mnet/{label_feature}/balanced'

    #     os.makedirs(orig_save_dir, exist_ok=True)
    #     os.makedirs(new_save_dir, exist_ok=True)

    #     print(f"balancing {label_feature}...")

    #     # -- Original dataloader
    #     orig_counts = count_samples(train_dataloader)
        
    #     plot_counts(orig_counts, label_feature, orig_save_dir)

    #     # -- Balanced dataloader
    #     balanced_train_dataloader = balance_samples(train_dataloader, label_feature)

    #     balanced_counts = count_samples(balanced_train_dataloader)

    #     plot_counts(balanced_counts, label_feature, True, new_save_dir)

    # plot_fonts()

    # plot_class_counts_mnet(train_dataloader, 'temperature')
    '''
    Input Chord Classes:
    - classic 84 chords let's gooooo
    
    Output Note Classes:
    - Notes: ['A#2', 'A#3', 'A#4', 'A#5', 'A#6', 'A2', 'A3', 'A4', 'A5', 'A6', 
              'B2', 'B3', 'B4', 'B5', 'B6', 'C#3', 'C#4', 'C#5', 'C#6', 'C#7', 
              'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'D#2', 'D#3', 'D#4', 'D#5', 'D#6', 'D#7', 
              'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 
              'F#2', 'F#3', 'F#4', 'F#5', 'F#6', 'F#7', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 
              'G#2', 'G#3', 'G#4', 'G#5', 'G#6', 'G#7', 'G2', 'G3', 'G4', 'G5', 'G6']
        * 459 total classes (note, octave, duration combinations)
    '''


      
    # train_dataloader, test_dataloader = create_dataloaders_mnet()

    # for song_inputs, song_labels in train_dataloader:
    #     print(f"song_inputs: {song_inputs[0][0]}, idx: {torch.argmax(song_inputs[0][0])}")
    #     print(f"song_labels: {song_labels[0][0]}")
    #     break



    # duration_counts = count_durations(labels_by_song)

    # Max duration: 64

    # -- Get all chord classes as encodings and strings
    # chord_classes = set()

    # for chord_inputs in inputs_by_song:
    #     for chord_input in chord_inputs:
    #         chord_classes.add(chord_input)

    # print(f"chord classes: {sorted(list(chord_classes))}\n")

    # chord_classes_strings = sorted(list(set([CHORD_ENC_TO_STR.get(chord_class) \
    #                              for chord_class in chord_classes])))
    
    # print(f"chord classes strings: {chord_classes_strings}, len: {len(chord_classes_strings)}\n")

    # # -- Get all note classes as encodings and strings
    # note_classes = set()
    # root_notes = set()

    # for note_labels in labels_by_song:
    #     for note_label in note_labels:
    #         note_classes.add(note_label)
    #         root_notes.add(note_label[:3])

    # sorted_note_classes = sorted(
    #     list(note_classes),
    #     key=lambda x: (int(x[:3]), int(x[3:]))
    # )
    
    # print(f"sorted note classes: {sorted_note_classes}\n")
    # print(len(note_classes))

    # root_notes = sorted(list(root_notes))

    # print(f"root notes: {root_notes}\n")

    # root_notes.remove('000')

    # root_notes_strings = sorted(list(set([NOTE_ENC_TO_NOTE_STR_REF.get(root_note[:2]) + str(root_note[2]) \
    #                           for root_note in root_notes])))
                    
    # print(f"root notes strings: {root_notes_strings}, len: {len(root_notes_strings)}\n")

    # # -- Create note to index mapping
    # note_classes_strings = sorted(list(set([NOTE_ENC_TO_NOTE_STR_REF.get(note_class[:2]) + note_class[2:] for note_class in note_classes])), key=sort_key)

    # print(f"note_classes_strings: {note_classes_strings}, len: {len(note_classes_strings)}")

    # note_str_to_idx = {}
    
    # for i, note_class in enumerate(note_classes_strings):
    #     note_str_to_idx[note_class] = i

    # with open('note_enc_to_idx.json', 'w') as file:
    #     json.dump(note_str_to_idx, file, indent=4)


    # # -- Create chord to index mapping
    # chord_str_to_idx = {}

    # for i, chord_class in enumerate(chord_classes_strings):
    #     chord_str_to_idx[chord_class] = i

    # with open('chord_enc_to_idx.json', 'w') as file:
    #     json.dump(chord_str_to_idx, file, indent=4)


    # # -- Plot note class distribution
    # total_note_labels = [note_label for note_labels in labels_by_song for note_label in note_labels]
    # note_class_counts = Counter(total_note_labels)

    # sorted_counts = [note_class_counts[note_class] for note_class in sorted_note_classes]

    # # Plot using matplotlib
    # plt.figure(figsize=(20, 10))
    # plt.bar(range(len(sorted_counts)), sorted_counts, color='skyblue')

    # # Customize the plot
    # plt.xlabel('Note Classes')
    # plt.ylabel('Counts')
    # plt.title('Note Class Distribution')

    # # Adjust layout to prevent clipping of tick-labels
    # plt.tight_layout()
    # plt.savefig('note_class_distribution.png')







    #         # In between
    #         else:
                

    #         # -- On a new note...
    #         if note_lifespan == '0':
    #             # Add current note/chord to samples (if not first)
    #             if timestep != 0:

    #                 # If both the current note and note_enc are rest notes, just increment duration
    #                 if current_note == '0000' and note_enc == '0000':
    #                     note_duration += 1
    #                 else:
    #                     # Update current note's lifespan to duration before storing
    #                     current_note = current_note[:3] + str(note_duration)

    #                     song_notes.append(current_note)
    #                     song_chords.append(current_chord)

    #                     # Update current note/chord to the next note
    #                     current_note = note_enc
    #                     current_chord = chord_enc

    #                     # Initialize duration to 1
    #                     note_duration = 1

    #             # Don't add first note
    #             else:
    #                 current_note = note_enc
    #                 current_chord = chord_enc

    #                 # Initialize duration to 1
    #                 note_duration = 1

    #         # -- Increment duration of sustained notes
    #         else:
    #             note_duration += 1

    #     inputs_by_song.append(song_chords)
    #     labels_by_song.append(song_notes)

    #     break

    
    # print(len(inputs_by_song))
    # print(len(labels_by_song))




            # -- Convert chord encoding to string
            # chord_str = NOTE_ENC_TO_NOTE_STR_REF.get(chord_enc[:2]) + CHORD_TYPE_IDX_TO_STR.get(chord_enc[2])

            # chord_enc_to_str[chord_enc] = chord_str

            # -- Convert note encoding to string

    # print(json.dumps(chord_enc_to_str, indent=4, sort_keys=True))









    # mnet_main()

    # train_dataloader, test_dataloader = create_dataloaders_mnet()

    # print(f"Min Label: {min([label for _, song_labels in train_dataloader for label in song_labels.squeeze(0)])}")
    # print(f"Max Label: {max([label for _, song_labels in train_dataloader for label in song_labels.squeeze(0)])}")


    # for song_inputs, song_labels in train_dataloader:
    #     print(f"song_inputs: {song_inputs}")
    #     print(f"song_labels: {song_labels}")
    #     break
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model_names = ['noise', 'noise_2']
        
    # for model_name in model_names:

    #     print(f"\n--- {model_name} ---\n")

    #     mnet = torch.load(f'saved_models/mnet/{model_name}/weights/epoch1.pth', map_location=device)

    #     output_fixed = mnet.output_fixed.weight

    #     print(f"output_fixed (hidden2 chord to output note): {output_fixed}")

    #     print(f"output_fixed sum: {torch.sum(output_fixed, dim=1)}")

    #     hidden2_from_chord = mnet.hidden2_from_chord.weight

    #     print(f"hidden2_from_chord (input chord to hidden2): {hidden2_from_chord}")

    # # -- Initialize noise_test model and see initial parameters
    # model_name = 'noise_test'

    # mnet = MelodyNet(hidden1_size=256, lr=0.05, weight_decay=0.0, repetition_weight=500.0,
    #                  chord_weight=20.0, melody_weight=5.0, state_units_decay=0.1, 
    #                  rest_fixed_weight=0.1, rest_loss_weight=1.0,
    #                  inject_noise=True, noise_size=100, noise_weight=1.0,
    #                  model_name=model_name)
    
    # print(f"\n--- {mnet.model_name} Pre-Training ---\n")

    # output_fixed = mnet.output_fixed.weight

    # print(f"output_fixed (hidden2 chord to output note): {output_fixed}")

    # print(f"output_fixed sum: {torch.sum(output_fixed, dim=1)}")

    # hidden2_from_chord = mnet.hidden2_from_chord.weight

    # print(f"hidden2_from_chord (input chord to hidden2): {hidden2_from_chord}")

    # # -- Train noise_test model and see post-training parameters
    # mnet_main()

    # mnet = torch.load(f'saved_models/mnet/{model_name}/weights/epoch1.pth', map_location=device)

    # print(f"\n--- {mnet.model_name} Post-Training ---\n")

    # output_fixed = mnet.output_fixed.weight

    # print(f"output_fixed (hidden2 chord to output note): {output_fixed}")

    # print(f"output_fixed sum: {torch.sum(output_fixed, dim=1)}")

    # hidden2_from_chord = mnet.hidden2_from_chord.weight

    # print(f"hidden2_from_chord (input chord to hidden2): {hidden2_from_chord}")

