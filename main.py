import torch
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
import pandas as pd
import os

from models.hnn import HNN
from utils.data.load_data import create_dataloaders, plot_class_counts
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

# -- hidden1_128
    hnn = HNN(hidden1_size=128, lr=0.05, weight_decay=1e-5,
              melody_weights=10.0, chord_weights=2.5, state_units_decay=0.75,
              model_name='hidden1_128')

# -- hidden1_128_state_05
    hnn = HNN(hidden1_size=128, lr=0.05, weight_decay=1e-5,
              melody_weights=10.0, chord_weights=2.5, state_units_decay=0.5,
              model_name='hidden1_128_state_05')

# -- hidden1_128_state_075
    hnn = HNN(hidden1_size=128, lr=0.05, weight_decay=1e-5,
              melody_weights=10.0, chord_weights=2.5, state_units_decay=0.75,
              model_name='hidden1_128_state_075')
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
    hnn = HNN(hidden1_size=256, lr=0.01, weight_decay=0.0,
              melody_weights=10.0, chord_weights=2.5, state_units_decay=0.75,
              model_name='huge_size_high_weights_low_lr')

    # -- Put model on device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hnn.to(device)

    # -- Define criterion/optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.SGD(hnn.parameters(), lr=hnn.lr, momentum=hnn.momentum, weight_decay=hnn.weight_decay)

    '''
    Training/Testing
    '''
    # -- Create model weights dir/model figs dir
    weights_dir = f'saved_models/hnn/{hnn.model_name}/weights'
    figs_dir = f'saved_models/hnn/{hnn.model_name}/figs'
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    epochs = 1000
    for epoch in range(1, epochs + 1):
        # -- Train for an epoch and store epoch loss
        epoch_loss = train(hnn, train_dataloader, criterion, optimizer, device)
        train_losses.append(epoch_loss)
        print(f"-- Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

        # -- Evaluate model on train set
        train_accuracy = test(hnn, train_dataloader, device)
        train_accuracies.append(train_accuracy * 100)
        print(f"Training Accuracy: {train_accuracy*100:.2f}%")

        # -- Evaluate model on test set
        test_accuracy = test(hnn, test_dataloader, device)
        test_accuracies.append(test_accuracy * 100)
        print(f"Testing Accuracy: {test_accuracy*100:.2f}%\n")

        # -- Save trained model every 10th epoch
        if epoch % 10 == 0:
            torch.save(hnn, f'{weights_dir}/epoch{epoch}.pth')

    '''
    Plot Training & Testing Loss/Accuracy
    '''
    # -- Plot training loss
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='red')

    plt.title(f'Train Loss ({hnn.model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{figs_dir}/train_loss.png')
    plt.close()

    # -- Plot training accuracy
    plt.figure()
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', color='green')

    plt.title(f'Train Accuracy ({hnn.model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{figs_dir}/train_accuracy.png')
    plt.close()

    # -- Plot testing accuracy
    plt.figure()
    plt.plot(range(1, epochs + 1), test_accuracies, label='Testing Accuracy', color='blue')

    plt.title(f'Test Accuracy ({hnn.model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{figs_dir}/test_accuracy.png')
    plt.close()

    # -- Coplot loss & accuracies
    plt.figure()

    # Plot training loss
    ax1 = plt.gca()
    ax1.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # Plot training/testing accuracy w/ shared x-axis
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', color='green')
    ax2.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='blue')
    ax2.set_ylabel('Accuracy (%)')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title(f'Train Loss & Train/Test Accuracy ({hnn.model_name})')
    plt.tight_layout()
    plt.savefig(f'{figs_dir}/train_test_loss_accuracy.png')
    plt.close()

    '''
    Write Training Metrics/Model Params to CSV
    '''
    metrics = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'test_accuracy': test_accuracies
    }

    params = {
        'hidden1_size': hnn.hidden1_size,
        'lr': hnn.lr,
        'weight_decay': hnn.weight_decay,
        'melody_weights': hnn.melody_weights,
        'chord_weights': hnn.chord_weights,
        'state_units_decay': hnn.state_units_decay
    }

    df = pd.DataFrame(metrics)
    df.to_csv(f'saved_models/hnn/{hnn.model_name}/metrics.csv', index=False)

    df = pd.DataFrame(params, index=[0])
    df.to_csv(f'saved_models/hnn/{hnn.model_name}/params.csv', index=False)


if __name__ == "__main__":
    train_dataloader, test_dataloader = create_dataloaders()

    plot_class_counts(test_dataloader)
    # main()