import pygame
import numpy as np
import time
from mappings import NOTE_FREQUENCIES, CHORD_NOTES

# -- Define constants for playback
SAMPLE_RATE = 44100
FADE_DURATION = 0.02
VOLUME = 0.25

NOTE_DURATION = 1
NOTE_OCTAVE = 4
CHORD_DURATION = 4
CHORD_OCTAVE = 4


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
    note_channel.play(pygame.sndarray.make_sound(note_wave))


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
    chord_channel.play(pygame.sndarray.make_sound(chord_wave))


def play_comp(notes, chord):
    '''
    Plays notes over a backing chord
    '''
    play_chord(chord)

    if notes[0] is not None:
        for note in notes:
            play_note(note)
            time.sleep(NOTE_DURATION)
    else:
        time.sleep(NOTE_DURATION)


def main():
    # -- Initialize Pygame mixer
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()

    notes = ['C#5']
    chord = 'Amaj'
    
    play_comp(notes, chord)

# if __name__ == "__main__":
#     main()
