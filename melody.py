import torch
import torch.nn.functional as F
import numpy as np
import pygame

from hnn import HNN
from load_data import create_training_data
from mappings import IDX_TO_NOTE_REF, IDX_TO_CHORD_REF
from play import *

def main():
    # -- Initialize Pygame mixer
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()

    # -- Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Load HNN model in eval mode
    hnn = HNN()
    hnn.load_state_dict(torch.load("hnn.pth", map_location=device, weights_only=True))

    hnn.eval()

    # -- Load song
    songs_inputs, songs_labels = create_training_data()
    song_input = songs_inputs[40]

    # -- Initialize state units
    state_units = torch.zeros((1, 14)).to(device)

    # -- For each timestep, pass melody input and playback chord output
    for timestep in range(len(song_input)):
        # Get one-hot encoded melody note at current timestep
        melody_input = song_input[timestep].unsqueeze(0)
        
        # Determine meter units
        meter_units = F.one_hot(torch.arange(2, dtype=torch.long))[timestep % 2].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
        meter_units = meter_units.expand((1, 2))

        # Concatenate state_units, melody inputs, and meter_units
        inputs = torch.cat([state_units, melody_input, meter_units], dim=1)
        
        # Forward pass
        output = hnn(inputs)

        # Update state units
        state_units = F.softmax(output)

        # Get note/chord as strings for playback
        note_idx = np.argwhere(melody_input.squeeze(0) == 1)
        if note_idx.numel() == 0:
            note = None
        else:
            note = IDX_TO_NOTE_REF.get(int(note_idx[0][0]))
        
        chord_idx = int(np.argmax(output.detach().squeeze(0)))
        chord = IDX_TO_CHORD_REF.get(chord_idx)

        # Playback note/chord
        print(f"note: {note}, chord: {chord}")
        play_comp([note], chord)


if __name__ == "__main__":
    main()