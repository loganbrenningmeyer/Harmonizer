import torch

'''
-- Model Architecture --

Input Layer:
- 28 total units:
    * 14 State Units
        - Softmax values of previous output
    * 12 Melody Units
        - One-hot encoding of the current melody note (chromatic scale)
    * 2 Meter Units
        - One-hot encoding of the current beat (1st or 3rd beat)

1st Hidden Layer:
- [12, 24] Neurons:
    * 

2nd Hidden Layer:
-

Output Layer:
-

'''