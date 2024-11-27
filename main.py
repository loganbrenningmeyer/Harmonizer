import torch
import torch.nn as nn
import torch.optim as opt
import os

from models.hnn import HNN
from utils.data.load_data import create_dataloaders
from utils.experiment import train, test


'''
Model Parameters:

# -- hidden1_64_melody_10
hnn = HNN(hidden1_size=64, lr=0.05, weight_decay=1e-4,
            melody_weights=10.0, chord_weights=2.5, state_units_decay=0.75,
            model_name='hidden1_64_melody_10')

# -- hidden1_64_melody_10_state_05
hnn = HNN(hidden1_size=64, lr=0.05, weight_decay=1e-4,
            melody_weights=10.0, chord_weights=2.5, state_units_decay=0.5,
            model_name='hidden1_64_melody_10_state_05')
'''


def main():
    '''
    Create DataLoaders
    '''
    train_dataloader, test_dataloader = create_dataloaders()

    '''
    Create Model
    '''
    # -- Initialize model
    hnn = HNN(hidden1_size=64, lr=0.05, weight_decay=1e-4,
              melody_weights=10.0, chord_weights=2.5, state_units_decay=0.5,
              model_name='hidden1_64_melody_10_state_05')

    # -- Put model on device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hnn.to(device)

    # -- Define criterion/optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.SGD(hnn.parameters(), lr=hnn.lr, momentum=hnn.momentum, weight_decay=hnn.weight_decay)

    '''
    Training
    '''
    # -- Create model saving directory
    os.makedirs(f'saved_models/hnn/{hnn.model_name}', exist_ok=True)

    epochs = 100
    for epoch in range(1, epochs + 1):
        epoch_loss = train(hnn, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

        if epoch % 10 == 0:
            # -- Save trained model
            torch.save(hnn, f'saved_models/hnn/{hnn.model_name}/epoch{epoch}.pth')

    '''
    Testing
    '''
    accuracy = test(hnn, test_dataloader, device)

    print(f"Testing Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()