'''
Contains mappings for one-hot encoding data from 
the chord-melody-dataset into a machine learning friendly format 
'''

'''
NOTES:
0.  11:     A
1.  12/20:  A#/Bb
2.  21:     B
3.  31:     C
4.  32/40:  C#/Db
5.  41:     D
6.  42/50:  D#/Eb
7.  51:     E
8.  61:     F
9.  62/70:  F#/Gb
10. 71:     G
11. 72/10:  G#/Ab
'''
NOTES_REF = {
    # -- A
    '11' : 0,
    # -- A#/Bb
    '12' : 1,
    '20' : 1,
    # -- B
    '21' : 2,
    # -- C
    '31' : 3,
    # -- C#/Db
    '32' : 4,
    '40' : 4,
    # -- D
    '41' : 5,
    # -- D#/Eb
    '42' : 6,
    '50' : 6,
    # -- E
    '51' : 7,
    # -- F
    '61' : 8,
    # -- F#/Gb
    '62' : 9,
    '70' : 9,
    # -- G
    '71' : 10,
    # -- G#/Ab
    '72' : 11,
    '10' : 11
}

'''
CHORDS:
0.  110: Amaj  
1.  210: Bmaj  
2.  310: Cmaj  
3.  410: Dmaj
4.  510: Emaj
5.  610: Fmaj
6.  710: Gmaj
7.  115: Adom7
8.  215: Bdom7
9.  315: Cdom7
10. 415: Ddom7
11. 515: Edom7
12. 615: Fdom7
13. 715: Gdom7
'''
CHORDS_REF = {
    # -- Amaj
    '110' : 0,
    # -- Bmaj
    '210' : 1,
    # -- Cmaj
    '310' : 2,
    # -- Dmaj
    '410' : 3,
    # -- Emaj
    '510' : 4,
    # -- Fmaj
    '610' : 5,
    # -- Gmaj
    '710' : 6,
    # -- Adom7
    '115' : 7,
    # -- Bdom7
    '215' : 8,
    # -- Cdom7
    '315' : 9,
    # -- Ddom7
    '415' : 10,
    # -- Edom7
    '515' : 11,
    # -- Fdom7
    '615' : 12,
    # -- Gdom7
    '715' : 13
}