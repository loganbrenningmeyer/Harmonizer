import pygame
import torch
import torch.nn.functional as F
import numpy as np
import time

from utils.data.mappings import *
from utils.data.load_data import get_songs_notes, get_songs_chords, parse_data

# -- Define constants for playback
SAMPLE_RATE = 44100
FADE_DURATION = 0.02
VOLUME = 0.25

NOTE_DURATION = 1
CHORD_DURATION = 8

# -- Active sounds list to ensure playback
active_sounds = []


def generate_sine_wave(frequency, duration, sample_rate=SAMPLE_RATE):
    """
    Generates a sine wave for a given frequency and duration with fade-in and fade-out.
    """
    # -- Create sine wave for given frequency, sample_rate, and duration
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(frequency * t * 2 * np.pi)
    
    # -- Apply linear fade-in/fade-out to avoid audio pops
    fade_length = int(sample_rate * FADE_DURATION)
    envelope = np.ones_like(wave)

    envelope[:fade_length] = np.linspace(0, 1, fade_length)
    envelope[-fade_length:] = np.linspace(1, 0, fade_length)
    
    wave *= envelope
    
    # -- Normalize to 16-bit range
    audio = wave * (32767 / np.max(np.abs(wave)))

    return audio.astype(np.int16)


def play_note(note, note_duration=NOTE_DURATION, volume=VOLUME):
    # -- Get note frequency
    note_freq = NOTE_FREQUENCIES[note]

    # -- Create sine wave w/ note frequency
    note_wave = generate_sine_wave(note_freq, note_duration).astype(np.float32)
    
    # -- Scale volume to avoid distortion
    note_wave *= (volume)

    # -- Clip to 16-bit range
    note_wave = np.clip(note_wave, -32767, 32767)

    # -- Convert to stereo as 16-bit int
    note_wave = np.column_stack((note_wave, note_wave)).astype(np.int16)

    # -- Play note on channel 1
    note_channel = pygame.mixer.Channel(1)
    note_sound = pygame.sndarray.make_sound(note_wave)
    note_channel.play(note_sound)

    # -- Append to active_sounds
    active_sounds.append(note_sound)

def play_chord(chord, chord_duration=CHORD_DURATION, volume=VOLUME):
    # -- Get chord note frequencies
    chord_note_freqs = [NOTE_FREQUENCIES[note] for note in CHORD_NOTES[chord]]

    # -- Initialize base wave
    chord_wave = np.zeros(int(SAMPLE_RATE * chord_duration), dtype=np.float32)

    # -- Add sine waves for each chord note frequency
    for note_freq in chord_note_freqs:
        note_wave = generate_sine_wave(note_freq, chord_duration).astype(np.float32)
        # Scale volume to avoid distortion
        chord_wave += note_wave * (volume / len(chord_note_freqs))

    # -- Clip to 16-bit range
    chord_wave = np.clip(chord_wave, -32767, 32767)

    # -- Convert to stereo as 16-bit int
    chord_wave = np.column_stack((chord_wave, chord_wave)).astype(np.int16)

    # -- Play chord on channel 0
    chord_sound = pygame.sndarray.make_sound(chord_wave)
    chord_channel = pygame.mixer.Channel(0)
    chord_channel.stop()
    chord_channel.play(chord_sound, loops=-1)

    # -- Append to active_sounds
    active_sounds.append(chord_sound)


def play_comp(notes, chord, note_duration: int = NOTE_DURATION):
    '''
    Plays notes over a backing chord
    '''
    play_chord(chord)

    if notes[0] is not None:
        for note in notes:
            play_note(note, note_duration)
            time.sleep(note_duration)
    else:
        time.sleep(note_duration)


def play_song_mnet(model_path: str, song_idx: int, note_duration: float,
                  topk: int = 0, multinomial: bool = False):
    # -- Initialize Pygame mixer
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    clock = pygame.time.Clock()

    print(f"\n-------- model: {model_path}, song_idx: {song_idx} --------\n")

    # -- Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Load MelodyNet model in eval mode
    mnet = torch.load(model_path, map_location=device)
    mnet.eval()

    # -- Get key/chord encodings of specified song_idx
    song_data = parse_data(hnn_data=False)[song_idx]
    song_chords = song_data['chords']
    song_key = song_data['key']
    # song_chords = get_songs_chords()[song_idx]

    print(f"Song Length: {len(song_chords)}")
    print(f"Song Key: {song_key}, Notes: {KEY_NOTES.get(song_key)}")

    # -- Initialize chord one-hot encoding arrays to index from
    chord_one_hot_array = torch.eye(mnet.chord_size, dtype=int)

    # -- Initialize state units
    state_units = torch.zeros((1, mnet.output_size)).to(device)

    current_chord_str = None

    timestep = 0

    while timestep < len(song_chords):
        # Get one-hot encoding chord index
        chord_enc = song_chords[timestep]
        # chord_idx = 7 * NOTE_ENC_TO_IDX_REF.get(chord[:2]) + int(chord[2])
        chord_str = CHORD_ENC_TO_STR.get(chord_enc)
        chord_idx = CHORD_STR_TO_IDX_MNET.get(chord_str)

        # Obtain chord input as one-hot encoding
        input_t = chord_one_hot_array[chord_idx].unsqueeze(0).float()

        # Determine meter units
        meter_units = F.one_hot(torch.arange(mnet.meter_size, dtype=torch.long))[timestep % mnet.meter_size].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
        meter_units = meter_units.expand((1, mnet.meter_size))

        # Inject noise if inject_noise == True
        if mnet.inject_noise:
            # noise = torch.randn((1, 100)).to(device)
            noise = torch.randn((1, mnet.noise_size)).to(device) * mnet.noise_weight
            inputs = torch.cat([state_units, input_t, noise, meter_units], dim=1)
        else:
            # print(f"{state_units.shape}, {input_t.shape}, {meter_units.shape}")
            inputs = torch.cat([state_units, input_t, meter_units], dim=1)

        # Forward pass
        output = mnet(inputs)

        # print(f"output class: {int(np.argmax(output.detach().squeeze(0)))}")

        # Update state units
        state_units = F.softmax(output / mnet.temperature, dim=1) + mnet.state_units_decay * state_units
        state_units = state_units / state_units.sum(dim=1, keepdim=True)

        # Get note as string for playback
        # chord_root = NOTE_ENC_TO_NOTE_STR_REF.get(chord_enc[:2])
        # chord_type = CHORD_TYPE_IDX_TO_STR.get(chord_enc[2])
        # chord_str = chord_root + chord_type

        # note_idx = int(np.argmax(output.detach().squeeze(0)))

        # Top-K sampling
        if topk > 0:
            output_probs = F.softmax(output / mnet.temperature, dim=1).squeeze()

            top_probs, top_indices = torch.topk(output_probs, topk)
            top_probs = top_probs / top_probs.sum()
            top_idx = torch.multinomial(top_probs, num_samples=1)

            note_idx = top_indices[top_idx].item()
        # Sample from output probabilities
        elif multinomial:
            output_probs = F.softmax(output / mnet.temperature, dim=1).squeeze()

            note_idx = torch.multinomial(output_probs, num_samples=1).item()
        # Take highest output class
        else:
            note_idx = int(np.argmax(output.detach().squeeze(0)))


        
        
        # if note_idx in range(0, 41):
        #     note_str = None
        #     note_dur = note_duration
        # else:
            # note_root_idx = (note_idx - 1) // 24
            # note_lifespan = ((note_idx - 1) - (note_root_idx * 24)) // 6
            # note_octave = ((note_idx - 1) - (note_root_idx * 24)) % 6 + 2

            # note_str = IDX_TO_NOTE_STR_REF.get(note_root_idx) + str(note_octave)

        note_idx_to_str = {v:k for k,v in NOTE_STR_TO_IDX_MNET.items()}
        note_str_label = note_idx_to_str[note_idx]

        if note_str_label[0] == 'R':
            note = 'R'
            note_str = None
            note_dur = int(note_str_label[2:])
        elif note_str_label[1] == '#':
            note = note_str_label[:2]
            note_str = note_str_label[:3]
            note_dur = int(note_str_label[3:])
        else:
            note = note_str_label[0]
            note_str = note_str_label[:2]
            note_dur = int(note_str_label[2:])

        print(f"chord: {chord_str}, note: {note_str}, duration: {note_dur}, in_key: {note in KEY_NOTES.get(song_key)}, harmonizes: {note in HARMONIZING_NOTES.get(chord_str)}")

        # Play chord when it changes
        if chord_str != current_chord_str:
            play_chord(chord_str, chord_duration=8 * note_duration)
            current_chord_str = chord_str

        if note_str is not None:
            play_note(note_str, note_dur * note_duration)
        
        # clock.tick(1 / (2 ** note_lifespan * note_duration))
        time.sleep(note_dur * note_duration)

        timestep += note_dur


def play_song_hnn(model_path: str, song_idx: int, note_duration: int = NOTE_DURATION):
    # -- Initialize Pygame mixer
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()

    print(f"\n-------- model: {model_path}, song_idx: {song_idx} --------\n")

    # -- Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Load HNN model in eval mode
    hnn = torch.load(model_path, map_location=device)
    hnn.eval()

    # -- Get note encodings of specified song_idx
    song_notes = get_songs_notes()[song_idx]

    # -- Initialize note one-hot encoding arrays to index from
    note_enc_array = torch.eye(hnn.melody_size, dtype=int)

    # -- Initialize state units
    state_units = torch.zeros((1, hnn.output_size)).to(device)

    for timestep in range(len(song_notes)):
        # Get one-hot encoded melody input
        note = song_notes[timestep]
        note_idx = NOTE_ENC_TO_IDX_REF.get(note[:2])

        if note_idx is not None:
            input_t = note_enc_array[note_idx].unsqueeze(0)
        else:
            input_t = torch.zeros((1, 12), dtype=int)

        # Determine meter units
        meter_units = F.one_hot(torch.arange(hnn.meter_size, dtype=torch.long))[timestep % hnn.meter_size].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
        meter_units = meter_units.expand((1, hnn.meter_size))

        # Concatenate state_units, melody inputs, and meter_units
        inputs = torch.cat([state_units, input_t, meter_units], dim=1)
        
        # Forward pass
        output = hnn(inputs)

        # Update state units
        state_units = F.softmax(output, dim=1) + hnn.state_units_decay * state_units
        state_units = state_units / state_units.sum(dim=1, keepdim=True)

        # Get note/chord as string for playback
        root_note = NOTE_ENC_TO_NOTE_STR_REF.get(note[:2])

        if root_note is not None and root_note != 'R':
            note_str = root_note + note[2]
        else:
            note_str = None

        chord_idx = int(np.argmax(output.detach().squeeze(0)))
        chord_str = IDX_TO_CHORD_STR_REF.get(chord_idx)

        # Playback note/chord
        print(f"note: {note_str}, chord: {chord_str}")
        play_comp([note_str], chord_str, note_duration)