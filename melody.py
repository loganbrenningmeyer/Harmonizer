import torch

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

print(fixed_output_weights)