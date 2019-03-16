# We will use this script to do any plots we might need for the report

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def main():
    problem_4_1()


def problem_4_1():
    lc_path = "absolute path to use"

    train_ppls, val_ppls, epochs_end_time = load_plot_values(lc_path)
    epochs_end_time = [x/60.0 for x in epochs_end_time]
    epochs = range(len(train_ppls))

    # Learning curves over epochs
    fig, ax = plt.subplots()
    ax.plot(epochs, train_ppls, 'k', label='Training')
    ax.plot(epochs, val_ppls, 'g', label='Validation')
    # plt.xticks(np.arange(min(epochs), max(epochs) + 1, 1.0))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PPL')
    ax.legend(loc='upper right', shadow=True, fontsize='large')
    plt.show()

    # Learning curves over wall-clock-time
    fig, ax = plt.subplots()
    ax.plot(epochs_end_time, train_ppls, 'k', label='Training')
    ax.plot(epochs_end_time, val_ppls, 'g', label='Validation')
    ax.set_xlabel('Wall-clock-time (minutes)')
    ax.set_ylabel('PPL')
    ax.legend(loc='upper right', shadow=True, fontsize='large')
    plt.show()


def load_plot_values(
    path
):
    """
    Load the data from an existing learning curves file at path

    :param path: Path of the learning_curves.npy to load
    :return: train_ppls, validation_ppls, epochs_end_time
    """
    values = np.load(path)[()]
    return values.get('train_ppls'), values.get('val_ppls'), values.get('epochs_end_time')


if __name__ == '__main__':
    main()
