import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
from scipy.stats import entropy, wasserstein_distance
import math

from utils.data.load_data import *
from utils.data.mappings import *
from utils.data.distributions import get_label_feature


def M_idx_to_enc(M_indices):
    '''
    Converts a melody from index representations to encodings
    '''
    M = []

    note_str_to_enc = {v:k for k,v in NOTE_ENC_TO_NOTE_STR_REF.items()}

    for m_t in M_indices:
        note_str = get_label_feature(m_t, 'notes')
        note_enc = note_str_to_enc[note_str]

        octave = str(get_label_feature(m_t, 'octaves'))

        duration = str(get_label_feature(m_t, 'durations'))

        m_t_enc = note_enc + octave + duration
        M.append(m_t_enc)

    return M


def C_idx_to_enc(C_indices):
    '''
    Converts a chord sequence from index representations to encodings
    '''
    C = []

    chord_idx_to_str = {v:k for k,v in CHORD_STR_TO_IDX_MNET.items()}
    chord_str_to_enc = {v:k for k,v in CHORD_ENC_TO_STR.items()}

    for c_t in C_indices:
        c_t_str = chord_idx_to_str[c_t]
        c_t_enc = chord_str_to_enc[c_t_str]

        C.append(c_t_enc)

    return C


def M_enc_to_str(M_enc):
    '''
    Converts a melody from its encoded representations to 
    note strings

    Parameters:
    - M_enc: Melody in encoded format (e.g., 6255)

    Returns:
    - M: Melody in string format (e.g., 'C')
    '''
    M = []

    for m_t in M_enc:
        M.append(NOTE_ENC_TO_NOTE_STR_REF.get(m_t[:2]))

    return M


def C_enc_to_str(C_enc):
    '''
    Converts a chord sequence from its encoded representations to 
    chord strings

    Parameters:
    - C_enc: Chord in encoded format (e.g., 625)

    Returns:
    - C: Chord in string format (e.g., 'Cmaj')
    '''
    C = []

    for c_t in C_enc:
        C.append(CHORD_ENC_TO_STR.get(c_t))

    return C



def intervals(M):
    '''
    Compute melody intervals for Structure

    Parameters:
    - M: List of melody note sequence over T timesteps 
         (formatted note encodings: note(2) oct(1) dur(:)

    Returns:
    - I: List of integer melody note intervals of size (T - 1) - num_rests 
    '''
    # -- Remove rests from interval calculation
    M = [m_t for m_t in M if m_t[:2] != '00']

    # -- Define chromatic note order (offsets)
    note_offsets = {
        'A': 0, 'A#': 1, 'B': 2, 'C': 3, 'C#': 4, 'D': 5, 
        'D#': 6, 'E': 7, 'F': 8, 'F#': 9, 'G': 10, 'G#': 11
    }

    # -- Initialize empty intervals array
    I = []

    # -- Initialize note position of previous note (m_t0)
    m_t0 = M[0]

    m_t0_note = NOTE_ENC_TO_NOTE_STR_REF.get(m_t0[:2])
    m_t0_octave = int(m_t0[2])
    m_t0_pos = note_offsets[m_t0_note] + 12 * m_t0_octave

    # -- Compute interval between notes t ∈ [1, T]
    for m_t in M[1:]:
        # -- Determine note position of current note (m_t)
        m_t_note = NOTE_ENC_TO_NOTE_STR_REF.get(m_t[:2])
        m_t_octave = int(m_t[2])
        m_t_pos = note_offsets[m_t_note] + 12 * m_t_octave

        # -- Compute interval (i_t = m_t+1 - m_t)
        i_t = m_t_pos - m_t0_pos
        I.append(i_t)

        # -- Update m_t0_pos to m_t_pos
        m_t0_pos = m_t_pos

    return I


def durations(M):
    '''
    Extract melody durations for Structure

    Parameters:
    - M: List of melody note sequence over T timesteps 
         (formatted note encodings: note(2) oct(1) dur(:)

    Returns:
    - D: List of integer melody note durations of size T
    '''
    D = []

    for m_t in M:
        D.append(int(m_t[3:]))

    return D


def LZW(M):
    '''
    Lempel-Ziv-Welch Compression Algorithm for Structure

    Parameters: 
    - M: List of melody note sequence over T timesteps (note encodings)

    Returns:
    - M_c: Compressed melody sequence
    '''
    # -- Convert note encodings to strings
    M = M_enc_to_str(M)

    # -- Initialize the dictionary with single symbols
    dictionary = {symbol: idx for idx, symbol in enumerate(set(M), start=1)}

    # -- Determine the next dictionary code idx
    next_code = max(dictionary.values()) + 1

    w = ""
    M_c = []

    for symbol in M:
        wk = w + symbol
        if wk in dictionary:
            w = wk
        else:
            M_c.append(dictionary[w])
            dictionary[wk] = next_code
            next_code += 1
            w = symbol

    # -- Add the last sequence
    if w:
        M_c.append(dictionary[w])

    return M_c


def harmoniousness(M, C, params):
    '''
    Harmoniousness
    - 
    => Q_\text{h}(M) = 1 - \frac{\sum_{t=1}^T P_\mathrm{h}(m_t,c_t)}{T \cdot P_\mathrm{h}^\mathrm{max}}
    
    
    -- Penalties
    => \begin{align*}P_\mathrm{h}^\mathrm{chord} &= 0 \\P_\mathrm{h}^\mathrm{scale} &= 2 \\P_\mathrm{h}^\mathrm{diss} &= 8 \\P_\mathrm{h}^\mathrm{max} &= T \cdot P_\mathrm{h}^\mathrm{diss}\end{align*}
    
    
    Parameters:
    - M: List of melody note sequence over T timesteps (note encodings)
    - C: List of chord sequence over T timesteps (chord encodings)

    Returns:
    - Q_h [0,1]: Harmoniousness heuristic value
    '''
    # -- Convert note/chord encodings to strings
    M = M_enc_to_str(M)
    C = C_enc_to_str(C)

    # -- Determine total timesteps T
    T = len(M)

    # -- Count each level of dissonance
    num_chord = num_scale = num_diss = 0

    # -- Define penalties by level of dissonance
    P_chord = params['Q_h']['P_chord']
    P_scale = params['Q_h']['P_scale']
    P_diss = params['Q_h']['P_diss']
    P_max = T * P_diss

    # -- Initialize total Harmoniousness penalty
    P_total = 0

    # -- Sum penalties over each melody/chord pair
    for i, (m_t, c_t) in enumerate(zip(M, C)):
        # Track previous note to account for tension/resolution
        m_t0 = m_t

        # Ignore rests
        if m_t == 'R':
            continue

        # -- Chord Tones
        if m_t in CHORD_TONES.get(c_t):
            # Account for resolving notes
            if i != 0:
                if m_t0 == 'scale':
                    P_total -= P_scale
                elif m_t0 == 'diss':
                    P_total -= P_diss / 2

            m_t0 = 'chord'
            num_chord += 1

        # -- Scale Non-Chord Tones
        elif m_t in HARMONIZING_NOTES.get(c_t):
            P_total += P_scale

            m_t0 = 'scale'
            num_scale += 1

        # -- Dissonant Tones
        else:
            P_total += P_diss

            m_t0 = 'diss'
            num_diss += 1

    # print(f"Harmoniousness --> P_total: {P_total}, P_max: {P_max}, P_total/P_max: {P_total/P_max}")

    # -- Take the complement of normalized P_total
    Q_h = 1 - (P_total / P_max)

    return Q_h, (num_chord, num_scale, num_diss)


def structure(M, params):
    '''
    Structure
    - 

    Compressibility
    => \mathrm{Comp}(M) = 1 - \frac{\vert \mathrm{LZW}(M) \vert}{\vert M \vert}

    Self-Similarity Matrices
    => S_I(a,b) = \exp\left(-\frac{(i_a - i_b)^2}{2\sigma^2}\right) \quad \forall \ a,b \in \{1,2,\ldots,T-1\}
    => S_D(a,b) = \delta_{d_a,d_b} \quad \forall \ a,b \in \{1,2,\ldots,T\}
    
    Recurrence Rate
    => RR_I = \frac{\vert \{(a,b) \ \vert \ 1 \leq a < b \leq T-1,\ S_I(a,b)=1\} \vert}{\frac{1}{2}(T-1)(T-2)} \quad \forall \ a,b \in \{1,2,\ldots,T-1\}
    => RR_D = \frac{\vert \{(a,b) \ \vert \ 1 \leq a < b \leq T,\ S_D(a,b)=1\} \vert}{\frac{1}{2}T(T-1)} \quad \forall \ a,b \in \{1,2,\ldots,T\}
    
    Probability Distribution
    => p_I(a,b) = \frac{S_I(a,b)}{\sum_{c=1}^{T-2} \sum_{d=c+1}^{T-1} S_I(c,d)} \quad \forall \ a,b \in \{1,2,\ldots,T-1\}
    => p_D(a,b) = \frac{S_D(a,b)}{\sum_{c=1}^{T-1} \sum_{d=c+1}^{T} S_D(c,d)} \quad \forall \ a,b \in \{1,2,\ldots,T\}
    
    Normalized Entropy
    => H_I^\text{norm} = \frac{-\sum_{a=1}^{T-2} \sum_{b=a+1}^{T-1} p_I(a,b) \log_2 p_I(a,b)}{log_2 (\frac{1}{2}(T-2)(T-1))} \quad \forall \ a,b \in \{1,2,\ldots,T-1\}
    => H_D^\text{norm} = \frac{-\sum_{a=1}^{T-1} \sum_{b=a+1}^{T} p_D(a,b) \log_2 p_D(a,b)}{log_2 (\frac{1}{2}(T-1)T)} \quad \forall \ a,b \in \{1,2,\ldots,T\}
    
    Weighted Sum of Structure Components
    => Q_\text{s}(M) = w_\mathrm{Comp} \cdot \mathrm{Comp}(M) + w_{RR_I} \cdot RR_I + w_{H_I} \cdot (1 - H_I^\text{norm}) + w_{RR_D} \cdot RR_D + w_{H_D} \cdot (1 - H_D^\text{norm})
    
    
    -- Structure Component Weights
    => \begin{align*}w_\mathrm{Comp} &= 0.2 \\w_{RR_I} &= 0.2 \\w_{H_I} &= 0.2 \\w_{RR_D} &= 0.2 \\w_{H_D} &= 0.2\end{align*}
    
    
    Parameters:
    - M: List of melody note sequence over T timesteps (note encodings)
    - sigma: Parameter for interval SSM Gaussian function

    Returns:
    - Q_s [0,1]: Structure heuristic value
    '''
    # -- Define Structure component weights
    w_Comp = params['Q_s']['w_Comp']
    w_RR_I = params['Q_s']['w_RR_I']
    w_RR_D = params['Q_s']['w_RR_D']
    w_H_I = params['Q_s']['w_H_I']
    w_H_D = params['Q_s']['w_H_D']

    RR_I_thresh = params['Q_s']['RR_I_thresh']

    sigma = params['Q_s']['sigma']

    # -- Define total timesteps T
    T = len(M)

    # -- Compressibility
    Comp_M = 1 - len(LZW(M)) / T

    # -- Determine melody intervals/durations
    I = intervals(M)
    D = durations(M)

    len_I = len(I)
    len_D = len(D)

    # -- Interval SSM (S_I)
    S_I = np.zeros((len_I, len_I))

    for a in range(len_I):
        for b in range(a + 1, len_I):
            S_I[a, b] = np.exp(-((I[a] - I[b]) ** 2) / (2 * sigma**2))

    # -- Duration SSM (S_D)
    S_D = np.zeros((len_D, len_D))

    for a in range(len_D):
        for b in range(a + 1, len_D):
            S_D[a, b] = 1 if D[a] == D[b] else 0

    # -- Recurrence Rates
    RR_I = np.sum(S_I >= RR_I_thresh) / (0.5 * len_I * (len_I - 1))
    RR_D = np.sum(S_D == 1) / (0.5 * len_D * (len_D - 1))

    # -- Flattened Probability Distributions
    p_I = np.triu(S_I, k=1) / np.sum(np.triu(S_I, k=1))
    p_D = np.triu(S_D, k=1) / np.sum(np.triu(S_D, k=1))

    p_I_flat = p_I[np.triu_indices(len_I, k=1)]
    p_D_flat = p_D[np.triu_indices(len_D, k=1)]

    # -- Normalized Entropy
    H_I = entropy(p_I_flat)
    H_D = entropy(p_D_flat)

    H_I_norm = H_I / math.log2(0.5 * len_I * (len_I - 1))
    H_D_norm = H_D / math.log2(0.5 * len_D * (len_D - 1))

    # print(f"Structure --> Comp: {Comp_M}, RR_I: {RR_I}, (1 - H_I): {1 - H_I_norm}, RR_D: {RR_D}, (1 - H_D): {1 - H_D_norm}")

    # -- Weighted Sum of Structure Components
    Q_s = w_Comp*Comp_M + w_RR_I*RR_I + w_RR_D*RR_D + w_H_I*(1 - H_I_norm) + w_H_D*(1 - H_D_norm)

    return Q_s, (Comp_M, RR_I, RR_D, 1 - H_I_norm, 1 - H_D_norm)


def balance(M, C, params):
    '''
    Balance
    - 

    Earth Mover's Distance (EMD)
    => \mathrm{EMD}_\mathrm{chord} = \mathrm{EMD}(p_\mathrm{chord}(M), p_\mathrm{chord}(X^*))
    => \mathrm{EMD}_\mathrm{scale} = \mathrm{EMD}(p_\mathrm{scale}(M), p_\mathrm{scale}(X^*))
    => \mathrm{EMD}_\mathrm{diss} = \mathrm{EMD}(p_\mathrm{diss}(M), p_\mathrm{diss}(X^*))
    
    Complement of the Weighted Sum of Normalized EMDs
    => Q_\mathrm{b}(M) = 1 - \frac{w_\mathrm{chord} \cdot \mathrm{EMD}_\mathrm{chord} + w_\mathrm{scale} \cdot \mathrm{EMD}_\mathrm{scale} + w_\mathrm{diss} \cdot \mathrm{EMD}_\mathrm{diss}}{b - a}
    
    
    -- Dissonance Level Weights
    => \begin{align*}w_\mathrm{chord} &= 0.33 \\w_\mathrm{scale} &= 0.33 \\w_\mathrm{diss} &= 0.33 \end{align*}
    
    
    Parameters: 
    - M: List of melody note sequence over T timesteps (note encodings)
    - C: List of chord sequence over T timesteps (note encodings)

    Returns:
    - Q_b [0,1]: Balance heuristic value 
    '''
    # -- Define Balance EMD weights
    w_chord = params['Q_b']['w_chord']
    w_scale = params['Q_b']['w_scale']
    w_diss = params['Q_b']['w_diss']

    norm_factor = 3

    # -- Convert melody/chords to strings
    M = M_enc_to_str(M)
    C = C_enc_to_str(C)

    # -- Establish note to histogram bin mapping
    note_indices = {
        'A' : 0, 'A#': 1, 'B' : 2, 'C' : 3, 'C#': 4, 'D' : 5,
        'D#': 6, 'E' : 7, 'F' : 8, 'F#': 9, 'G' : 10,'G#': 11
    }

    # -- Initialize empty probability distributions
    p_chord_tones = torch.zeros(12, dtype=torch.float32)
    p_scale_tones = torch.zeros(12, dtype=torch.float32)
    p_diss_tones = torch.zeros(12, dtype=torch.float32)

    # -- Histograms by Dissonance Level
    for m_t, c_t in zip(M, C):
        # -- Ignore rests
        if m_t == 'R':
            continue

        # -- Determine note index
        note_idx = note_indices[m_t]

        # -- Chord Tones
        if m_t in CHORD_TONES.get(c_t):
            p_chord_tones[note_idx] += 1

        # -- Scale Non-Chord Tones
        elif m_t in HARMONIZING_NOTES.get(c_t):
            p_scale_tones[note_idx] += 1

        # -- Dissonant Tones
        else:
            p_diss_tones[note_idx] += 1

    # -- Sort Histograms
    p_chord_tones = p_chord_tones.sort(descending=True)[0]
    p_scale_tones = p_scale_tones.sort(descending=True)[0]
    p_diss_tones = p_diss_tones.sort(descending=True)[0]

    # -- Convert to Probability Distributions
    total_chord_tones = torch.sum(p_chord_tones)
    total_scale_tones = torch.sum(p_scale_tones)
    total_diss_tones = torch.sum(p_diss_tones)

    if total_chord_tones != 0:
        p_chord_tones /= total_chord_tones
    if total_scale_tones != 0:
        p_scale_tones /= total_scale_tones
    if total_diss_tones != 0:
        p_diss_tones /= total_diss_tones

    # -- Load "Ideal" Probability Distributions
    df = pd.read_csv('dataset_dists.csv')
    dataset_dists = df.to_dict(orient='list')

    # -- Convert distributions to numpy arrays
    p_ideal_chord_tones = np.array(dataset_dists['Chord'])
    p_ideal_scale_tones = np.array(dataset_dists['Scale'])
    p_ideal_diss_tones = np.array(dataset_dists['Dissonant'])

    p_chord_tones = p_chord_tones.cpu().numpy()
    p_scale_tones = p_scale_tones.cpu().numpy()
    p_diss_tones = p_diss_tones.cpu().numpy()

    # -- Compute Earth Mover's Distance (Wasserstein Distance)
    pos = np.arange(12)


    if np.sum(p_chord_tones) != 0:
        EMD_chord = wasserstein_distance(pos, pos, p_chord_tones, p_ideal_chord_tones)
    else:
        EMD_chord = 2.5

    if np.sum(p_scale_tones) != 0:
        EMD_scale = wasserstein_distance(pos, pos, p_scale_tones, p_ideal_scale_tones)
    else:
        EMD_scale = 2.5

    if np.sum(p_diss_tones) != 0:
        EMD_diss = wasserstein_distance(pos, pos, p_diss_tones, p_ideal_diss_tones)
    else:
        EMD_diss = 2.5


    # print(f"Balance --> EMD_chord: {EMD_chord}, EMD_scale: {EMD_scale}, EMD_diss: {EMD_diss}")

    # -- Complement of the Weighted Sum of Normalized EMDs
    Q_b = 1 - ((w_chord*EMD_chord + w_scale*EMD_scale + w_diss*EMD_diss) / norm_factor)

    return Q_b, (EMD_chord / norm_factor, EMD_scale / norm_factor, EMD_diss / norm_factor)


def flow(M, params):
    '''
    Flow
    - 
    
    Interval Penalties
    => P_\mathrm{f}(i_t) = \begin{cases} P_\mathrm{f}^\mathrm{unison}, & \text{if } i_t = 0 \\ P_\mathrm{f}^\mathrm{stepwise}, & \text{if } 1 \leq i_t \leq 2 \\ P_\mathrm{f}^\mathrm{conjunct}, & \text{if } 3 \leq i_t \leq 5 \\ P_\mathrm{f}^\mathrm{disjunct} & \text{if } i_t \geq 6 \end{cases}

    Normalized Sum of Penalties
    => Q_\mathrm{f}(M) = 1 - \frac{\sum_{t=1}^{T-1} P_\mathrm{f}(i_t)}{(T - 1) \cdot P_\mathrm{f}^\mathrm{max}}
    
    
    -- Assigned Interval Penalties
    => \begin{align*}P_\mathrm{f}^\mathrm{unison} &= 10 \\P_\mathrm{f}^\mathrm{stepwise} &= 1 \\P_\mathrm{f}^\mathrm{conjunct} &= 3 \\P_\mathrm{f}^\mathrm{disjunct} &= 4 \cdot (i_t - 5) \\P_\mathrm{f}^\mathrm{max} &= 4 \cdot (12 - 5) = 28\end{align*}
    
    
    Parameters:
    - M: List of melody note sequence over T timesteps (note encodings)

    Returns:
    - Q_f: Flow heuristic value
    '''
    # -- Store counts for each interval
    num_unison = num_stepwise = num_conjunct = num_disjunct = 0

    # -- Define Interval Penalties
    P_unison = params['Q_f']['P_unison']
    P_stepwise = params['Q_f']['P_stepwise']
    P_conjunct = params['Q_f']['P_conjunct']

    def P_disjunct(i_t):
        return min(params['Q_f']['P_disjunct'] * (i_t - 5), P_unison)
    
    # -- Determine Intervals
    I = intervals(M)
    
    # -- Define Max Penalty
    P_max = len(I) * P_unison
    
    # -- Compute Total Interval Penalty
    P_total = 0

    for i_t in I:
        # -- Unison
        if i_t == 0:
            P_total += P_unison
            num_unison += 1

        # -- Stepwise Motion
        elif 1 <= abs(i_t) <= 2:
            P_total += P_stepwise
            num_stepwise += 1

        # -- Conjunct Leap
        elif 3 <= abs(i_t) <= 5:
            P_total += P_conjunct
            num_conjunct += 1

        # -- Disjunct Leap
        else:
            P_total += P_disjunct(abs(i_t))
            num_disjunct += 1

    # print(f"Flow --> P_total: {P_total}, P_max: {P_max}, P_total/P_max: {P_total/P_max}")

    # -- Complement of Normalized Total Penalty
    Q_f = 1 - (P_total / P_max)

    return Q_f, (num_unison, num_stepwise, num_conjunct, num_disjunct, P_total, P_max)


def Q(M, C, params):
    '''
    Melody Quality
    - 

    => Q(M) = w_\text{h} \cdot Q_\text{h}(M) + w_\text{s} \cdot Q_\text{s}(M) + w_\text{b} \cdot Q_\text{b}(M) + w_\text{f} \cdot Q_\text{f}(M)
    
    
    -- Melody Quality Component Weights
    => \begin{align*}w_\mathrm{h} &= 0.4 \\w_\mathrm{s} &= 0.2 \\w_\mathrm{b} &= 0.2 \\w_\mathrm{f} &= 0.2\end{align*}
    '''
    # -- Define Melody Quality Component Weights
    w_h = params['Q_M']['w_h']
    w_s = params['Q_M']['w_s']
    w_b = params['Q_M']['w_b']
    w_f = params['Q_M']['w_f']

    Q_h, Q_h_metrics = harmoniousness(M, C, params)
    Q_s, Q_s_metrics = structure(M, params)
    Q_b, Q_b_metrics = balance(M, C, params)
    Q_f, Q_f_metrics = flow(M, params)

    Q_M = w_h*Q_h + w_s*Q_s + w_b*Q_b + w_f*Q_f

    heuristics = {
        'Q_M': Q_M,
        'Q_h': [Q_h, Q_h_metrics],
        'Q_s': [Q_s, Q_s_metrics],
        'Q_b': [Q_b, Q_b_metrics],
        'Q_f': [Q_f, Q_f_metrics]
    }

    # print(f"Heuristics --> Q(M): {Q_M}, Q_h: {Q_h}, Q_s: {Q_s}, Q_b: {Q_b}, Q_f: {Q_f}\n")

    return heuristics


def plot_model_comparison(paths: list[str], output_dir: str):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    splits = ['train', 'test']
    combined_dfs = {}

    # Process train and test
    for split in splits:
        dfs = []
        model_names = []

        for path in paths:
            df = pd.read_csv(os.path.join(path, 'eval', f'{split}_heuristics_eval.csv'))
            model_name = os.path.basename(path)
            model_names.append(model_name)
            df['Model Name'] = model_name
            dfs.append(df)

        # Combine all models for this split
        dfs = pd.concat(dfs, ignore_index=True)
        combined_dfs[split] = dfs

        # Define the metrics to compare
        metrics = [r"Q_M", r"Q_h", r"Q_s", r"Q_b", r"Q_f"]
        metric_labels = {
            "Q_M": r"$Q_M$",
            "Q_h": r"$Q_h$",
            "Q_s": r"$Q_s$",
            "Q_b": r"$Q_b$",
            "Q_f": r"$Q_f$"
        }

        # Set colors
        palette = sns.color_palette("Set2", n_colors=len(paths))
        group_colors = {m: palette[i] for i, m in enumerate(model_names)}

        # Create and save plots for this split
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            ax = sns.histplot(
                data=dfs,
                x=metric,
                hue='Model Name',
                palette=group_colors,
                bins=50,
                kde=False,
                edgecolor='black',
                alpha=0.6
            )
            plt.title(f"{metric_labels[metric]}", fontsize=16, fontweight='bold')
            plt.xlabel("Value", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            legend = ax.get_legend()
            handles = legend.legend_handles
            ax.legend(handles, model_names)
            plt.grid(True)

            # Save plot
            plt.savefig(os.path.join(split_output_dir, f'{split}_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot for '{metric}' at '{os.path.join(split_output_dir, f'{split}_{metric}.png')}'")

    # Now create a "full" dataset by combining train and test
    full_df = pd.concat([combined_dfs['train'], combined_dfs['test']], ignore_index=True)

    # Use the same metrics and plotting approach for the full dataset
    full_output_dir = os.path.join(output_dir, 'full')
    os.makedirs(full_output_dir, exist_ok=True)

    metrics = [r"Q_M", r"Q_h", r"Q_s", r"Q_b", r"Q_f"]
    metric_labels = {
        "Q_M": r"$Q_M$",
        "Q_h": r"$Q_h$",
        "Q_s": r"$Q_s$",
        "Q_b": r"$Q_b$",
        "Q_f": r"$Q_f$"
    }

    # The models and palette can remain consistent as before
    model_names = full_df['Model Name'].unique()
    palette = sns.color_palette("Set2", n_colors=len(model_names))
    group_colors = {m: palette[i] for i, m in enumerate(model_names)}

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(
            data=full_df,
            x=metric,
            hue='Model Name',
            palette=group_colors,
            bins=50,
            kde=False,
            edgecolor='black',
            alpha=0.6
        )
        plt.title(f"{metric_labels[metric]} (Full Dataset)", fontsize=16, fontweight='bold')
        plt.xlabel("Value", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        legend = ax.get_legend()
        handles = legend.legend_handles
        ax.legend(handles, model_names)
        plt.grid(True)

        # Save plot
        plt.savefig(os.path.join(full_output_dir, f'full_{metric}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved full dataset plot for '{metric}' at '{os.path.join(full_output_dir, f'full_{metric}.png')}'")



def plot_grouped_heuristics(model_path, output_dir='heuristic_plots'):
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.join(model_path, ), exist_ok=True)
    
    # Load the data
    data = pd.read_csv(file_path)
    
    # Define the groups and subplots structure
    groupings = {
        "Overall Melody Quality": {
            "main": ["Q_M"],
            "subgroups": []
        },
        "Harmoniousness": {
            "main": ["Q_h"],
            "subgroups": [["num_chord", "num_scale", "num_diss"]]
        },
        "Structure": {
            "main": ["Q_s", "Comp_M"],
            "subgroups": [["RR_I", "RR_D"], ["H_I_norm", "H_D_norm"]]
        },
        "Balance": {
            "main": ["Q_b"],
            "subgroups": [["EMD_chord", "EMD_scale", "EMD_diss"]]
        },
        "Flow": {
            "main": ["Q_f"],
            "subgroups": [["num_unison", "num_stepwise", "num_conjunct", "num_disjunct"]]
        }
    }

    # Custom labeling for metrics
    metric_labels = {
        # Main Q metrics
        "Q_M": r"Overall Melody Quality ($Q_M$)",
        "Q_h": r"Harmoniousness ($Q_h$)",
        "Q_s": r"Structure ($Q_s$)",
        "Q_b": r"Balance ($Q_b$)",
        "Q_f": r"Flow ($Q_f$)",

        # Harmoniousness subgroup
        "num_chord": "Chord Tones",
        "num_scale": "Scale Non-Chord Tones",
        "num_diss": "Dissonant Tones",

        # Structure metrics
        "Comp_M": r"Compressibility ($Comp(M)$)",
        "RR_I": r"Interval ($RR_I$)",
        "RR_D": r"Duration ($RR_D$)",
        "H_I_norm": r"Interval $(1 - H_{I}^{norm})$",
        "H_D_norm": r"Duration $(1 - H_{D}^{norm})$",

        # Balance metrics
        "EMD_chord": "Chord Tones",
        "EMD_scale": "Scale Non-Chord Tones",
        "EMD_diss": "Dissonant Tones",

        # Flow metrics
        "num_unison": "Unison Intervals (0)",
        "num_stepwise": "Stepwise Motion (1-2)",
        "num_conjunct": "Conjunct Leaps (3-5)",
        "num_disjunct": "Disjunct Leaps (6+)"
    }

    # Titles for specific subgroups
    subgroup_titles = {
        "num_chord,num_scale,num_diss": "Levels of Dissonance",
        "num_unison,num_stepwise,num_conjunct,num_disjunct": "Note Intervals",
        "RR_I,RR_D": "Recurrence Rate",
        "H_I_norm,H_D_norm": "Complement of Normalized Entropy",
        "EMD_chord,EMD_scale,EMD_diss": "Earth Mover's Distance Distributions"
    }

    # Select numeric columns
    heuristics_data = data.select_dtypes(include=['float64', 'int64']).copy()

    # Set plot style
    sns.set(style="whitegrid")

    for group_name, metrics in groupings.items():

        main_metrics = metrics["main"]
        subgroups = metrics["subgroups"]
        
        # Determine the number of subplots
        num_main = len(main_metrics)
        num_subgroups = len(subgroups)
        total_subplots = num_main + num_subgroups
        
        if total_subplots == 1:
            cols = 1
        else:
            cols = 2

        rows = (total_subplots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5), constrained_layout=True)
        
        # Flatten axes for consistency
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        fig.suptitle(f"Distribution of {group_name} Metrics", fontsize=18, fontweight='bold')
        
        plot_idx = 0
        
        # Plot main metrics (single metric subplots)
        for main_metric in main_metrics:
            ax = axes[plot_idx]
            # Single metric plot with KDE, no legend
            sns.histplot(
                heuristics_data[main_metric], 
                bins=50, 
                kde=True, 
                color='skyblue', 
                edgecolor='black', 
                ax=ax, 
                alpha=0.8
            )
            
            # Set the title with proper labeling
            title_label = metric_labels.get(main_metric, main_metric)
            ax.set_title(title_label, fontsize=14, fontweight='bold')
            
            ax.set_xlabel("Value", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.grid(True)
            
            plot_idx += 1
        
        # Plot subgroups (multi-metric subplots)
        for subgroup in subgroups:
            ax = axes[plot_idx]
            subgroup_data = heuristics_data[subgroup]
            melted = subgroup_data.melt(var_name='Metric', value_name='Value')
            melted['Metric_label'] = melted['Metric'].map(metric_labels)
            melted['Metric_label'] = melted['Metric_label'].fillna(melted['Metric'])

            sns.histplot(
                data=melted,
                x='Value',
                hue='Metric_label',
                hue_order=melted['Metric_label'].unique(),
                multiple='layer',
                kde=False,
                bins='auto',  # Use automatic binning
                palette='Set2',
                edgecolor='black', # Keep edge colors
                ax=ax,
                alpha=0.6 # Increased transparency to help when there is a lot of overlap
            )

            legend = ax.get_legend()
            handles = legend.legend_handles
            ax.legend(handles, [metric_labels[sub] for sub in subgroup])

            # Determine title based on subgroup
            subgroup_key = ",".join(subgroup)
            title_text = subgroup_titles.get(subgroup_key, "Metric Distributions")
            ax.set_title(title_text, fontsize=14, fontweight='bold')
            
            if group_name in ['Balance', 'Structure']:
                ax.set_xlabel("Value", fontsize=12)
            else:
                ax.set_xlabel("Count", fontsize=12)

            ax.set_ylabel("Frequency", fontsize=12)
            ax.grid(True)
            
            plot_idx += 1
        
        # Remove any unused subplots
        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])
        
        # Save the figure
        sanitized_group_name = group_name.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f"{sanitized_group_name}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot for '{group_name}' at '{output_path}'")


def main():
    '''
    All functions passed notes/chords in encoding formats,
    if necessary functions will handle string conversion themselves
    '''
    paths = ['saved_models/mnet/high_weights_comp/heuristics_eval.csv',
             'saved_models/mnet/med_weights_comp/heuristics_eval.csv',
             'saved_models/mnet/low_weights_comp/heuristics_eval.csv']
    plot_model_comparison(paths, output_dir='figs')
    # plot_grouped_heuristics('saved_models/mnet/low_weights_comp/heuristics_eval.csv', 
    #                         output_dir='saved_models/mnet/low_weights_comp/figs')
    



def dataset_heuristics():
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

    notes, chords = parse_formatted_data(strings=False)
    num_songs = len(notes)

    for M, C in zip(notes, chords):
        heuristics = Q(M, C)

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
        num_unison, num_stepwise, num_conjunct, num_disjunct = heuristics['Q_f'][1]

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

    '''
    Write Heuristics Training Values to csv
    '''
    heuristics = {
        # -- Melody Quality
        'Q_M': Q_M_vals,
        # -- Harmoniousness
        'Q_h': Q_h_vals,
        'num_chord': num_chord_vals,
        'num_scale': num_scale_vals,
        'num_diss': num_diss_vals,
        # -- Structure
        'Q_s': Q_s_vals,
        'Comp_M': Comp_M_vals,
        'RR_I': RR_I_vals,
        'RR_D': RR_D_vals,
        'H_I_norm': H_I_norm_vals,
        'H_D_norm': H_D_norm_vals,
        # -- Balance
        'Q_b': Q_b_vals,
        'EMD_chord': EMD_chord_vals,
        'EMD_scale': EMD_scale_vals,
        'EMD_diss': EMD_diss_vals,
        # -- Flow
        'Q_f': Q_f_vals,
        'num_unison': num_unison_vals,
        'num_stepwise': num_stepwise_vals,
        'num_conjunct': num_conjunct_vals,
        'num_disjunct': num_disjunct_vals
    }

    df = pd.DataFrame(heuristics)
    df.to_csv(f'data_heuristics.csv')

    # Q_h_stats = []
    # P_total_stats = []
    # P_max_stats = []

    # Q_M_stats = []
    # Q_h_stats = []
    # Q_s_stats = []
    # Q_b_stats = []
    # Q_f_stats = []

    # for i, (M, C) in enumerate(zip(notes, chords)):

    #     # Q_b, EMD_chord, EMD_scale, EMD_diss = balance(M, C)
    #     # Q_h, P_total, P_max = harmoniousness(M, C)

    #     # P_max_stats.append(P_max)
    #     # Q_h_stats.append(Q_h)
    #     # P_total_stats.append(P_total)

    #     # Optional: Print progress every 100 songs
    #     if (i + 1) % 100 == 0 or (i + 1) == num_songs:
    #         print(f"Processed {i + 1}/{num_songs} songs.")

    #     # Q_b_avg += Q_b
    #     # EMD_chord_avg += EMD_chord
    #     # EMD_scale_avg += EMD_scale
    #     # EMD_diss_avg += EMD_diss

    #     Q_M, Q_h, Q_s, Q_b, Q_f = Q(M, C)

    #     Q_M_stats.append(Q_M)
    #     Q_h_stats.append(Q_h)
    #     Q_s_stats.append(Q_s)
    #     Q_b_stats.append(Q_b)
    #     Q_f_stats.append(Q_f)

    # def print_stats(name, data):
    #     mean = np.mean(data)
    #     std_dev = np.std(data)
    #     min_val = np.min(data)
    #     max_val = np.max(data)
    #     print(f"--- {name} Statistics ---")
    #     print(f"Mean: {mean}")
    #     print(f"Standard Deviation: {std_dev}")
    #     print(f"Min: {min_val}")
    #     print(f"Max: {max_val}\n")

    # # Q_h_stats = np.array(Q_h_stats)
    # # P_total_stats = np.array(P_total_stats)
    # # P_max_stats = np.array(P_max_stats)

    # Q_M_stats = np.array(Q_M_stats)
    # Q_h_stats = np.array(Q_h_stats)
    # Q_s_stats = np.array(Q_s_stats)
    # Q_b_stats = np.array(Q_b_stats)
    # Q_f_stats = np.array(Q_f_stats)

    # print_stats('Q_h', Q_h_stats)
    # print_stats('Q_s', Q_s_stats)
    # print_stats('Q_b', Q_b_stats)
    # print_stats('Q_f', Q_f_stats)
    # print_stats('Q_M', Q_M_stats)
    # print_stats('P_total', P_total_stats)
    # print_stats('P_max', P_max_stats)

    # Q_b_avg /= num_songs
    # EMD_chord_avg /= num_songs
    # EMD_scale_avg /= num_songs
    # EMD_diss_avg /= num_songs

    # print(f"Q_b")

    # print(f"Q_b: {Q_b_avg}, EMD_chord: {EMD_chord_avg}, EMD_scale: {EMD_scale_avg}, EMD_diss: {EMD_diss_avg}")

    # print(f"Avg Q(M): {Q_M_avg / num_songs}")
    

if __name__ == "__main__":
    main()





# if __name__ == "__main__":
#     songs = parse_data(hnn_data=False)

#     print(songs[0]['notes'])
#     print(songs[0]['chords'])


# if __name__ == "__main__":
#     data_heuristics()


# def data_heuristics():
#     '''
#     Compute heuristics for the ground truth dataset, plotting distributions
#     and recording metrics for each
#     '''
#     songs = parse_data(hnn_data=False)

#     print(songs[0]['notes'])
#     print(songs[0]['chords'])





'''
----------------- What We Gon Do Babyyy!! ----------------- ✅

- Actually doing:
    1. Plot dataset's distribution of Melody Quality heuristics [...]
    2. Run the best models from different approaches over the dataset
       and plot their distributions as well
    3. Compare the averages between models and with the dataset, as well 
       as their distributions [ ]
        - Their distributions may help capture how consistent they are
          between different contexts
    4. Plot each heuristic's reward over the RL phase
        - See if the model leans into any heuristics in particular,
          and in what proportion

    5. For each heuristic, plot its submetrics over training
        - There must certainly be some metrics that are dominating the others
        - Structure spikes rapidly after the first iteration, causing flow
          to stagnate as it simply repeats notes

- Paper Focus, things to hit [ ]
    * Purpose behind my metrics
    * Justification of my metrics
        - Empirically measure on dataset
        - Plot distribution of dataset's songs vs my models' songs
            * Supervised Only
            * Supervised + Reinforcement
            * Reinforcement Only
        - See if the dataset is more consistent or my models are
            * See which heuristics they score higher on
        
    * Overall Goal:
        - Explain why standard machine learning metrics are not adequate
          for evaluating the performance of MelodyNet/MelodyNetRL
        - Therefore, I devised general melody quality heuristics to evaluate the 
          performance of my model from a more musically grounded standpoint
        - In addition, developing these heuristics allowed me to guide the models'
          training to align with more musical ideas
            * Building off HNN's idea of a "Human Neural Network"---the idea of 
              incorporating human knowledge into the neural network
            * The supervised loss signal is simply far too noisy given the 
              nature of the dataset and the inherent uncertainty of music
            * To combat this, guiding the model via RL was a natural solution to 
              incorporating external musical information---imbuing some human nature
              into the model
        - Comparison of the different model approaches:
            * Approaches
                - Pure Supervised Learning
                - SL/RL
                - Pure Reinforcement Learning
                - "Creativity" Parameters (maybe?) 
        - Comparison Methods
            * Melody Quality Heuristics
                - Biased to RL models
            * Dissonance Ratio
            * Pitch Class Accuracy
            * Harmoniziation Rate
            * Raw Accuracy

- Implement Melody Quality Heuristics
    * Harmoniousness [✅]
    * Structure [✅]
    * Balance [✅]
    * Flow [✅]
    * Melody Quality [✅]

- Tune Melody Quality Heuristics [☑️..]
    * Need more variety, probably normalize by smaller values
        - Improved Harmoniousness range
        - Improved Balance range
        - Fixed Flow negative issue
    * Look into average dissonant tones/harmonizing tones ratio
    * Look into maximum values to determine normalization
        - Maximum interval in the dataset
        - Maximum EMD_chord, EMD_scale, EMD_diss
          
- Supervised Learning & Reinforcement Learning Plan [✅]
    * Keep SL and RL separate; use one exclusively at a time
    * Pretrain using supervised learning (whole epoch)
        - Use ground truth notes for meter index
    * Fine-tune using reinforcement learning (whole epoch)
        - Use predicted notes for meter index
            * Chord inputs will have to be at 16th-beat granularity;
              you can't know where the predicted notes will end

- Reinforcement Learning Approach [✅]
    * At each iteration within a song, forward pass through the model then sample from its output softmax distribution
      to obtain its action a_t (note class) and its corresponding probability π_θ(a_t|s_t) (softmax probability)
    * Store each action's note class (melody sequence M) and corresponding probability (π_θ(a_t|s_t) for t in T)
    * Use the duration of the action (note class) to progress the timestep
    * At the end of the song (when timestep > song duration), compute the model's reward Q(M)
    * Compute REINFORCE loss by taking the sum of the log of all action probabilities, multiply by Q(M), and negate
    * Backward pass and update model with optimizer using REINFORCE loss, then repeat for the next song

- Update State Units [✅]
    * Change state units to output class feature embedding representations
    * State unit inputs are the last n predicted classes as feature embeddings
    * e.g., last 8 predictions * 32 length feature embeddings = 256 state units
    * use torch.nn.Embedding
        - vocab_size = 459
        - embedding_dim = 64 (or something smaller maybe like 32)
        - embedding_layer = nn.Embedding(vocab_size, embedding_dim)     
            * creates a mapping from [0, 458] to 64-dim feature vectors
        - note_indices = torch.tensor([0, 1, 2,..., 458])
        - note_vectors = embedding_layer(note_indices)   
            * outputs shape: [vocab_size, embedding_dim] or [459, 64]

- Analyze Dataset Distributions [✅]
    * Sorted Note Distributions for Levels of Dissonance
        - For each note per song, determine level of dissonance and add to bin
        - Sort the distributions
        - Pad missing bins (<12) with 0s
        - Average all distributions for dataset, save, and plot

- Evaluate Dataset Using Heuristics [✅..]
    * Find Q(M) for each song
    * Rank songs based on Q(M) and each component heuristic
        - If time allows, compare with real rankings
    * Compare dataset quality heuristics with generated melodies
        - Dataset may perform better, but hopefully close

- Play Dataset Songs [...]
    * Add a function for playback of dataset songs


------------- What am I putting in my report? -------------

- Discussion of the original HNN architecture and my implementation

- Explain the MelodyNet architecture
    * Influence from HNN architecture
    * Differences in melody generation vs chord generation
    * Explanation of architectural changes
        - State Units
            * Output probs (HNN) --> Class embeddings of past notes (MelodyNet)
                - Evaluate difference in performance for SL Only
            * State units decay
        - Additional Learnable Weights
            * hidden2 --> output
        - "Creativity" Parameters
            * temperature
            * dropout
            * repetition loss
    * Explanation of the impact of class balancing
        - Class
        - Pitch Class
        - Octave
        - Pitch Class/Octave
        - Duration

- Definitions and reasonings behind Melody Quality heuristics
    * Why They're Necessary
        - Dataset accuracy not indicative of model's ability
    * Purpose of Each
        - Harmoniousness
            * Melodies must harmonize with their chords
        - Structure
            * Melodies should have motifs, not randomness
        - Balance
            * Melodies typically focus on subsets of notes
        - Flow
            * Melodies should have smooth melodic contour
    * Explanation of Parameters

- Evaluation of the integrity of Melody Quality heuristics (may be hard, but probably good to do :( sad)
    * Find Q(M) of each song in the dataset
    * Rank songs by Q(M) and compare with their actual ratings/rankings

- Comparison of MelodyNet approaches using heuristics/metrics
    * Approaches
        - Supervised Learning
        - Supervised & Reinforcement Learning
        - Reinforcement Learning
    * Possible Unbiased Metrics
        - Interval distribution matching to the test set (EMD)
        - Duration distribution matching to the test set (EMD)
        - 1 - Proportion of dissonant notes
            * 1 - (# of dissonant notes / # of total notes)  -- higher is better

        - Pitch Class distribution matching to the test set (EMD)
            * Probably biased due to Balance heuristic
    * Parameter Considerations
        - Do not balance distributions, evaluate how heuristics affect
          the model's adherence to dataset's distributions
        - Do not use custom losses (repetition, harmony, key, etc.)
        - Universally usable parameters are acceptable:
            * Dropout
            * Chord/Melody Weights


--------------------- REINFORCE ---------------------

- Loss: ( - sum_t=1-->T [∇_θ log (π_θ(a_t | s_t))] ) * Q(M)
    * where:
        - a_t = the model's action at timestep t (sampled from softmax)
        - s_t = the state at timestep t (i.e., chord input, state units, meter units)
        - π_θ(a_t | s_t) = the probability of the action a_t given s_t (softmax probability of sampled action)
        - log() simplifies the multiplication of probabilities into a sum, easier computationally/mathematically
        - ∇_θ(log()) = gradient of log probability of a_t w/ respect to parameters θ (handled automatically by autograd)
        - Q(M) melody quality heuristic weights the gradients

- Possible improvements:
    * Use a running average for the reward baseline to reduce variance for more stable training
    * Scale rewards by normalizing with the running average and a running standard deviation

'''

# def main():
#     '''
#     Processes notes and chords to calculate Q_b and various EMD averages.
#     After processing all songs, it computes and prints the mean, standard deviation,
#     minimum, and maximum for each metric.
#     '''
#     # Parse the data to get notes and chords
#     notes, chords = parse_formatted_data(strings=False)
#     num_songs = len(notes)

#     # Initialize separate lists to store the metrics for each song
#     Q_b_avg = []
#     EMD_chord_avg = []
#     EMD_scale_avg = []
#     EMD_diss_avg = []

#     # Iterate over each song's notes and chords
#     for i, (M, C) in enumerate(zip(notes, chords)):
#         # Balance function returns Q_b and various EMD averages for the song
#         Q_b, EMD_chord, EMD_scale, EMD_diss = balance(M, C)

#         # Append the metrics to their respective lists
#         Q_b_avg.append(Q_b)
#         EMD_chord_avg.append(EMD_chord)
#         EMD_scale_avg.append(EMD_scale)
#         EMD_diss_avg.append(EMD_diss)

#         # Optional: Print progress every 100 songs
#         if (i + 1) % 100 == 0 or (i + 1) == num_songs:
#             print(f"Processed {i + 1}/{num_songs} songs.")

#     # Convert lists to numpy arrays for efficient computation
#     Q_b_avg = np.array(Q_b_avg)
#     EMD_chord_avg = np.array(EMD_chord_avg)
#     EMD_scale_avg = np.array(EMD_scale_avg)
#     EMD_diss_avg = np.array(EMD_diss_avg)

#     # Define a helper function to compute and print statistics
#     def print_stats(name, data):
#         mean = np.mean(data)
#         std_dev = np.std(data)
#         min_val = np.min(data)
#         max_val = np.max(data)
#         print(f"--- {name} Statistics ---")
#         print(f"Mean: {mean}")
#         print(f"Standard Deviation: {std_dev}")
#         print(f"Min: {min_val}")
#         print(f"Max: {max_val}\n")

#     # Compute and print statistics for each metric
#     print_stats("Q_b_avg", Q_b_avg)
#     print_stats("EMD_chord_avg", EMD_chord_avg)
#     print_stats("EMD_scale_avg", EMD_scale_avg)
#     print_stats("EMD_diss_avg", EMD_diss_avg)