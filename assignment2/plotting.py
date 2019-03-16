# We will use this script to do any plots we might need for the report

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def main():
    lc_path = "E:/Repos/ift6135-rnn/assignment2/experiments/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=200_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_num_epochs=5_save_best_save_dir=experiments_0/learning_curves.npy"
    train_ppls, val_ppls, epochs_end_time = load_plot_values(lc_path)

    fig, ax = plt.subplots()

    ax.plot(x_values, y_values, 'b-', label='plot label')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_title("Plot title")
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
