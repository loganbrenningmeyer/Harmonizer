import pygame
import torch
import torch.nn.functional as F
import numpy as np
import time

from utils.data.mappings import *
from utils.data.load_data import get_songs_notes, get_songs_chords

# -- Define constants for playback
SAMPLE_RATE = 44100
FADE_DURATION = 0.02
VOLUME = 0.25

NOTE_DURATION = 1
CHORD_DURATION = 4

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


def play_note(note, duration=NOTE_DURATION, volume=VOLUME):
    # -- Get note frequency
    note_freq = NOTE_FREQUENCIES[note]

    # -- Create sine wave w/ note frequency
    note_wave = generate_sine_wave(note_freq, duration).astype(np.float32)
    
    # -- Scale volume to avoid distortion
    note_wave *= (2 * volume / 2)

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

def play_chord(chord, duration=CHORD_DURATION, volume=VOLUME):
    # -- Get chord note frequencies
    chord_note_freqs = [NOTE_FREQUENCIES[note] for note in CHORD_NOTES[chord]]

    # -- Initialize base wave
    chord_wave = np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)

    # -- Add sine waves for each chord note frequency
    for note_freq in chord_note_freqs:
        note_wave = generate_sine_wave(note_freq, duration).astype(np.float32)
        # Scale volume to avoid distortion
        chord_wave += note_wave * (2 * volume / len(chord_note_freqs))

    # -- Clip to 16-bit range
    chord_wave = np.clip(chord_wave, -32767, 32767)

    # -- Convert to stereo as 16-bit int
    chord_wave = np.column_stack((chord_wave, chord_wave)).astype(np.int16)

    # -- Play chord on channel 0
    chord_channel = pygame.mixer.Channel(0)
    chord_sound = pygame.sndarray.make_sound(chord_wave)
    chord_channel.play(chord_sound)

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


def play_song_mnet(model_path: str, song_idx: int, note_duration: int = NOTE_DURATION):
    # -- Initialize Pygame mixer
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()

    print(f"\n-------- model: {model_path}, song_idx: {song_idx} --------\n")

    # -- Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Load MelodyNet model in eval mode
    mnet = torch.load(model_path, map_location=device)
    mnet.eval()

    # -- Get chord encodings of specified song_idx
    song_chords = get_songs_chords()[song_idx]

    print(len(song_chords))

    # -- Initialize chord one-hot encoding arrays to index from
    chord_enc_array = torch.eye(mnet.chord_size, dtype=int)

    # -- Initialize state units
    state_units = torch.zeros((1, mnet.output_size)).to(device)

    for timestep in range(len(song_chords)):
        print(f"timestep: {timestep}")
        # Get one-hot encoding chord index
        chord = song_chords[timestep]
        chord_idx = 7 * NOTE_ENC_TO_IDX_REF.get(chord[:2]) + int(chord[2])
        # Obtain chord input as one-hot encoding
        input_t = chord_enc_array[chord_idx].unsqueeze(0)

        # Determine meter units
        meter_units = F.one_hot(torch.arange(mnet.meter_size, dtype=torch.long))[timestep % mnet.meter_size].to(device)     # [1, 0] on 1st beat, [0, 1] on 3rd beat
        meter_units = meter_units.expand((1, mnet.meter_size))

        # Concatenate state_units, melody inputs, and meter_units
        inputs = torch.cat([state_units, input_t, meter_units], dim=1)

        # Forward pass
        output = mnet(inputs)

        # Update state units
        state_units = F.softmax(output, dim=1) + mnet.state_units_decay * state_units
        state_units = state_units / state_units.sum(dim=1, keepdim=True)

        # Get note/chord as strings for playback
        chord_root = NOTE_ENC_TO_NOTE_STR_REF.get(chord[:2])
        chord_type = CHORD_TYPE_IDX_TO_STR.get(chord[2])
        chord_str = chord_root + chord_type

        note_idx = int(np.argmax(output.detach().squeeze(0)))
        
        if note_idx == 0:
            note_str = None
        else:
            note_root_idx = (note_idx - 1) // 12
            note_lifespan = ((note_idx - 1) - (note_root_idx * 12)) // 6
            note_octave = ((note_idx - 1) - (note_root_idx * 12)) + 2

            note_str = IDX_TO_NOTE_STR_REF.get(note_root_idx) + str(note_octave)

        print(f"chord: {chord_str}, note: {note_str}")
        play_comp([note_str], chord_str, note_duration)


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
        if NOTE_ENC_TO_NOTE_STR_REF.get(note[:2]) is not None:
            note_str = NOTE_ENC_TO_NOTE_STR_REF.get(note[:2]) + note[2]
        else:
            note_str = None

        chord_idx = int(np.argmax(output.detach().squeeze(0)))
        chord_str = IDX_TO_CHORD_STR_REF.get(chord_idx)

        # Playback note/chord
        print(f"note: {note_str}, chord: {chord_str}")
        play_comp([note_str], chord_str, note_duration)