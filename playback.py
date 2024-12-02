from models.hnn import HNN
from models.mnet import MelodyNet
from utils.play import play_song_hnn, play_song_mnet
import random

NOTE_DURATION_HNN = 0.75
NOTE_DURATION_MNET = 0.2

'''
Run from Command Prompt to support audio output

avg_accuracy
-- Model: high_size_low_weights
-- Epoch: 553 (~epoch550.pth)
-- Accuracy: 41.635769362310725

train_accuracy
-- Model: huge_size_med_weights
-- Epoch: 715 (~epoch710.pth & ~epoch720.pth)
-- Accuracy: 48.70309050772627

test_accuracy
-- Model: high_size_high_weights_low_lr
-- Epoch: 538 (~epoch540.pth)
-- Accuracy: 39.61267605633803
'''

def hnn_main():
    '''
    Model Paths
    '''
    best_avg = 'saved_models/hnn/high_size_low_weights/weights/epoch550.pth'
    best_train = 'saved_models/hnn/huge_size_med_weights/weights/epoch710.pth'
    best_test = 'saved_models/hnn/high_size_high_weights_low_lr/weights/epoch540.pth' # 67 not bad

    # -- Low (1st Hidden Size: 32)
    low_high_low = 'saved_models/hnn/low_size_high_weights_low_lr/weights/epoch780.pth'
    low_low = 'saved_models/hnn/low_size_low_weights/weights/epoch190.pth'

    # -- Medium (1st Hidden Size: 64)
    med_high_low = 'saved_models/hnn/med_size_high_weights_low_lr/weights/epoch440.pth'
    med_high = 'saved_models/hnn/med_size_high_weights/weights/epoch400.pth'
    med_low = 'saved_models/hnn/med_size_low_weights/weights/epoch400.pth' # 57 pretty good

    # -- High (1st Hidden Size: 128)
    high_high = 'saved_models/hnn/high_size_high_weights/weights/epoch240.pth'
    high_med = 'saved_models/hnn/high_size_med_weights/weights/epoch200.pth'

    # -- Huge (1st Hidden Size: 256)
    huge_high_low = 'saved_models/hnn/huge_size_high_weights_low_lr/weights/epoch860.pth'

    '''
    Pick Random Song and Playback (79 HNN compatible songs)
    '''
    song_idx = random.randint(0, 78)

    play_song_hnn(model_path=best_test,
                  song_idx=67,
                  note_duration=NOTE_DURATION_HNN)
    

def mnet_main():
    '''
    Model Paths
    '''
    test = 'saved_models/mnet/reduced_rest/weights/epoch1.pth'
    repetition_penalty = 'saved_models/mnet/repetition_penalty/weights/epoch1.pth'
    duration = 'saved_models/mnet/duration/weights/epoch1.pth'
    duration_learnable = 'saved_models/mnet/duration_learnable/weights/epoch1.pth'

    high_rep_low_lr = 'saved_models/mnet/high_rep_low_lr/weights/epoch1.pth'
    huge_rep_low_lr = 'saved_models/mnet/huge_rep_low_lr/weights/epoch1.pth'
    huge_size_high_rep_low_lr = 'saved_models/mnet/huge_size_high_rep_low_lr/weights/epoch1.pth'
    huge_size_huge_rep_low_lr = 'saved_models/mnet/huge_size_huge_rep_low_lr/weights/epoch1.pth'
    huge_size_huge_weights = 'saved_models/mnet/huge_size_huge_weights/weights/epoch3.pth'

    low_size_huge_rep_low_rest = 'saved_models/mnet/low_size_huge_rep_low_rest/weights/epoch1.pth'
    noise = 'saved_models/mnet/noise/weights/epoch1.pth'

    low_size_chord_noise = 'saved_models/mnet/low_size_chord_noise/weights/epoch1.pth'
    high_size_chord_noise = 'saved_models/mnet/high_size_chord_noise/weights/epoch1.pth'
    huge_size_chord_noise = 'saved_models/mnet/huge_size_chord_noise/weights/epoch1.pth'
    low_size_chord_noise_half = 'saved_models/mnet/low_size_chord_noise_half/weights/epoch1.pth'
    low_size_noise_input = 'saved_models/mnet/low_size_noise_input/weights/epoch1.pth'
    low_size_noise_input_no_rest = 'saved_models/mnet/low_size_noise_input_no_rest/weights/epoch1.pth'
    low_size_noise_input_no_rest_med_rep = 'saved_models/mnet/low_size_noise_input_no_rest_med_rep/weights/epoch1.pth'

    # template = 'saved_models/mnet/_/weights/epoch1.pth'
    noise_weight = 'saved_models/mnet/noise_weight/weights/epoch2.pth'
    reduced_rest_loss = 'saved_models/mnet/reduced_rest_loss/weights/epoch1.pth'
    noise_high_mel_high_noise = 'saved_models/mnet/noise_high_mel_high_noise/weights/epoch1.pth'
    noise_2 = 'saved_models/mnet/noise_2/weights/epoch1.pth'
    noise_test = 'saved_models/mnet/noise_test/weights/epoch1.pth'

    low_weights = 'saved_models/mnet/low_weights/weights/epoch1.pth'
    dropout = 'saved_models/mnet/dropout/weights/epoch1.pth'
    temperature = 'saved_models/mnet/temperature/weights/epoch1.pth'
    rest_loss = 'saved_models/mnet/rest_loss/weights/epoch1.pth'

    new_data = 'saved_models/mnet/new_data/weights/epoch1.pth'
    new_data_take2 = 'saved_models/mnet/new_data_take2/weights/epoch1.pth'
    new_data_high_size = 'saved_models/mnet/new_data_high_size/weights/epoch1.pth'
    new_data_high_state = 'saved_models/mnet/new_data_high_size/weights/epoch1.pth'
    new_data_low_weights = 'saved_models/mnet/new_data_low_weights/weights/epoch1.pth'
    new_data_no_dropout = 'saved_models/mnet/new_data_no_dropout/weights/epoch1.pth'
    new_data_no_temp = 'saved_models/mnet/new_data_no_temp/weights/epoch1.pth'
    new_data_noise = 'saved_models/mnet/new_data_noise/weights/epoch1.pth'
    new_data_med_repetition = 'saved_models/mnet/new_data_med_repetition/weights/epoch1.pth'
    new_data_high_repetition = 'saved_models/mnet/new_data_high_repetition/weights/epoch1.pth'
    new_data_low_size = 'saved_models/mnet/new_data_low_size/weights/epoch1.pth'
    new_data_low_size_med_repetition = 'saved_models/mnet/new_data_low_size_med_repetition/weights/epoch1.pth'

    duration_test = 'saved_models/mnet/duration_test/weights/epoch1.pth'
    duration_test_high_weights = 'saved_models/mnet/duration_test_high_weights/weights/epoch6.pth'


    classes_balance = 'saved_models/mnet/classes_balance/weights/epoch1.pth'
    notes_balance = 'saved_models/mnet/notes_balance/weights/epoch1.pth'
    octaves_balance = 'saved_models/mnet/octaves_balance/weights/epoch1.pth'
    notes_octaves_balance = 'saved_models/mnet/notes_octaves_balance/weights/epoch1.pth'
    durations_balance = 'saved_models/mnet/durations_balance/weights/epoch1.pth'

    notes_octaves_balance_high_mel_rep_low_lr = 'saved_models/mnet/notes_octaves_balance_high_mel_rep_low_lr/weights/epoch1.pth' # 1816
    notes_octaves_no_fixed_high_rep_low_lr = 'saved_models/mnet/notes_octaves_no_fixed_high_rep_low_lr/weights/epoch1.pth'

    no_fixed_chords_low_lr = 'saved_models/mnet/no_fixed_chords_low_lr/weights/epoch1.pth'
    no_fixed_melody_low_lr = 'saved_models/mnet/no_fixed_melody_low_lr/weights/epoch1.pth'
    no_fixed_chords_melody_low_lr = 'saved_models/mnet/no_fixed_chords_melody_low_lr/weights/epoch1.pth'
    both_chords_melody_low_lr = 'saved_models/mnet/both_chords_melody_low_lr/weights/epoch1.pth'
    
    no_chords = 'saved_models/mnet/no_chords/weights/epoch1.pth'
    no_melody = 'saved_models/mnet/no_melody/weights/epoch1.pth'
    no_chords_no_melody = 'saved_models/mnet/no_chords_no_melody/weights/epoch1.pth'
    chords_and_melody = 'saved_models/mnet/chords_and_melody/weights/epoch1.pth'
    
    '''
    Pick Random song and playback (5167 MelodyNet compatible songs)
    '''
    song_idx = random.randint(0, 5166)

    play_song_mnet(
        model_path=notes_octaves_balance_high_mel_rep_low_lr,
        song_idx=1900,
        note_duration=NOTE_DURATION_MNET,
        topk=0,
        multinomial=False
    )

    
if __name__ == "__main__":
    # hnn_main()
    mnet_main()