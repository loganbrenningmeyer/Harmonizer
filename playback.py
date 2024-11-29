from models.hnn import HNN
from models.mnet import MelodyNet
from utils.play import play_song_hnn, play_song_mnet
import random

NOTE_DURATION_HNN = 0.75
NOTE_DURATION_MNET = 0.1

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

    # -- Medium (1st Hidden Size: )
    med_high_low = 'saved_models/hnn/med_size_high_weights_low_lr/weights/epoch440.pth'
    med_high = 'saved_models/hnn/med_size_high_weights/weights/epoch400.pth'
    med_low = 'saved_models/hnn/med_size_low_weights/weights/epoch400.pth' # 57 pretty good

    # -- High
    high_high = 'saved_models/hnn/high_size_high_weights/weights/epoch240.pth'
    high_med = 'saved_models/hnn/high_size_med_weights/weights/epoch200.pth'

    # -- Huge
    huge_high_low = 'saved_models/hnn/huge_size_high_weights_low_lr/weights/epoch860.pth'

    '''
    Pick Random Song and Playback (79 HNN compatible songs)
    '''
    song_idx = random.randint(0, 78)

    play_song_hnn(model_path=best_test,
                  song_idx=song_idx,
                  note_duration=NOTE_DURATION_HNN)
    

def mnet_main():
    '''
    Model Paths
    '''
    test = 'saved_models/mnet/test/weights/epoch1.pth'

    '''
    Pick Random song and playback (5167 MelodyNet compatible songs)
    '''
    song_idx = random.randint(0, 5166)

    play_song_mnet(model_path=test,
                   song_idx=song_idx,
                   note_duration=NOTE_DURATION_MNET)

    
if __name__ == "__main__":
    mnet_main()