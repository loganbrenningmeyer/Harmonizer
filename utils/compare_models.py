import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import os

def compare_plots():
    saved_models_dir = 'saved_models/hnn'

    fontname = 'Lato'
    fig_color = '#ffffe6'

    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    fig.set_facecolor(fig_color)
    axes = axes.flatten()

    size_labels = {
        'huge': 'Huge (256)',
        'high': 'High (128)',
        'med': 'Medium (64)',
        'low': 'Low (32)'
    }

    model_dirs = [
        ['huge_size_low_weights', 'huge_size_med_weights', 'huge_size_high_weights', 'huge_size_high_weights_low_lr'],
        ['high_size_low_weights', 'high_size_med_weights', 'high_size_high_weights', 'high_size_high_weights_low_lr'],
        ['med_size_low_weights', 'med_size_med_weights', 'med_size_high_weights', 'med_size_high_weights_low_lr'],
        ['low_size_low_weights', 'low_size_med_weights', 'low_size_high_weights', 'low_size_high_weights_low_lr']
    ]

    model_dirs = [model_dir for model_size in model_dirs for model_dir in model_size]

    for i, model_dir in enumerate(model_dirs):
        img = mpimg.imread(os.path.join(saved_models_dir, model_dir, 'figs', 'train_test_loss_accuracy.png'))

        axes[i].imshow(img)
        axes[i].set_title(model_dir, fontname=fontname, fontsize=14)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        if i % 4 == 0:
            axes[i].set_ylabel(size_labels[model_dir.split('_')[0]], fontsize=14, fontname=fontname)
        

    fig.suptitle('Model Comparison: Training Loss & Training/Testing Accuracy', y=0.93, fontsize=18, fontname=fontname)

    fig.text(0.07, 0.5, '1st Hidden Layer Size', va='center', rotation='vertical', fontsize=14, fontname=fontname, fontweight='bold')

    fig.savefig('model_comparison.png', bbox_inches='tight', dpi=300)


def best_epochs_accuracy():
    saved_models_dir = 'saved_models/hnn'

    model_dirs = os.listdir(saved_models_dir)
    model_dirs.remove('old')

    # -- Best overall model_name/epoch/accuracy for avg, train and test
    global_best = {
        'avg_accuracy': {'model': None, 'epoch': None, 'accuracy': 0.0},
        'train_accuracy': {'model': None, 'epoch': None, 'accuracy': 0.0},
        'test_accuracy': {'model': None, 'epoch': None, 'accuracy': 0.0},
    }

    # -- Best epoch/accuracy for each model (1-indexed)
    best_epochs = {}

    for model_dir in model_dirs:
        # -- Load metrics.csv
        df = pd.read_csv(os.path.join(saved_models_dir, model_dir, 'metrics.csv'))

        train_accuracy = df['train_accuracy'].to_numpy()
        test_accuracy = df['test_accuracy'].to_numpy()

        # -- Epoch with maximum average train/test accuracy
        avg_accuracy = (train_accuracy + test_accuracy) / 2

        best_epoch_avg = np.argmax(avg_accuracy)

        # -- Epoch with maximum train accuracy
        best_epoch_train = np.argmax(train_accuracy)

        # -- Epoch with maximum test accuracy
        best_epoch_test = np.argmax(test_accuracy)

        # -- Update global_best avg/train/test accuracy
        if avg_accuracy[best_epoch_avg] > global_best['avg_accuracy']['accuracy']:
            global_best['avg_accuracy'] = {
                'model': model_dir,
                'epoch': best_epoch_avg + 1,
                'accuracy': avg_accuracy[best_epoch_avg]
            }

        if train_accuracy[best_epoch_train] > global_best['train_accuracy']['accuracy']:
            global_best['train_accuracy'] = {
                'model': model_dir,
                'epoch': best_epoch_train + 1,
                'accuracy': train_accuracy[best_epoch_train]
            }

        if test_accuracy[best_epoch_test] > global_best['test_accuracy']['accuracy']:
            global_best['test_accuracy'] = {
                'model': model_dir,
                'epoch': best_epoch_test + 1,
                'accuracy': test_accuracy[best_epoch_test]
            }

        # -- Store best epochs in dictionary
        best_epochs[model_dir] = {
            'avg_accuracy': {
                'epoch': best_epoch_avg + 1,
                'accuracy': avg_accuracy[best_epoch_avg]
            },
            'train_accuracy': {
                'epoch': best_epoch_train + 1,
                'accuracy': train_accuracy[best_epoch_train]
            },
            'test_accuracy': {
                'epoch': best_epoch_test + 1,
                'accuracy': test_accuracy[best_epoch_test]
            }
        }

    print("-------- GLOBAL BEST --------")
    for metric, values in global_best.items():
        print(metric)
        print(f"-- Model: {values['model']}")
        print(f"-- Epoch: {values['epoch']}")
        print(f"-- Accuracy: {values['accuracy']}\n\n")

    print("-------- MODEL BEST --------")
    for model, values in best_epochs.items():
        avg_accuracy = values['avg_accuracy']
        train_accuracy = values['train_accuracy']
        test_accuracy = values['test_accuracy']

        print(model)
        print(f"-- Avg Accuracy: Epoch {avg_accuracy['epoch']}, {avg_accuracy['accuracy']}%")
        print(f"-- Train Accuracy: Epoch {train_accuracy['epoch']}, {train_accuracy['accuracy']}%")
        print(f"-- Test Accuracy: Epoch {test_accuracy['epoch']}, {test_accuracy['accuracy']}%\n")

    return global_best, best_epochs