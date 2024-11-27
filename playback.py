from models.hnn import HNN
from utils.play import play_song

'''
Run from Command Prompt to support audio output
'''

def main():
    play_song(model_path='saved_models/hnn/hidden1_64_melody_10/epoch100.pth', 
              song_idx=40)
    
if __name__ == "__main__":
    main()