# We will use this script to do any plots we might need for the report

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def main():
    problem_4_1_figures()


def problem_4_1_figures():
    results = [
        ["experiments_problem4_1/experiment_1_970gtx/learning_curves.npy", "Experiment 1", 0],
        ["experiments_problem4_1/experiment_2_970gtx/learning_curves.npy", "Experiment 2", 0],
        ["experiments_problem4_1/experiment_3_1070ti/learning_curves.npy", "Experiment 3", 2]
    ]

    for result in results:
        problem_4_1_draw_figure(file_path=result[0], figure_title=result[1], start_epoch=result[2])


def problem_4_1_draw_figure(file_path, figure_title, start_epoch):
    train_ppls, val_ppls, epochs_end_time = load_plot_values(file_path)
    train_ppls = train_ppls[start_epoch:]
    val_ppls = val_ppls[start_epoch:]
    epochs_end_time = [x / 60.0 for x in epochs_end_time[start_epoch:]]
    epochs = range(start_epoch, start_epoch + len(train_ppls))

    # Learning curves over epochs
    fig, ax = plt.subplots()
    ax.set_title(figure_title)
    ax.plot(epochs, train_ppls, 'k', label='Training')
    ax.plot(epochs, val_ppls, 'g', label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PPL')
    ax.legend(loc='upper right', shadow=True, fontsize='large')
    plt.show()

    # Learning curves over wall-clock-time
    fig, ax = plt.subplots()
    ax.set_title(figure_title)
    ax.plot(epochs_end_time, train_ppls, 'k', label='Training')
    ax.plot(epochs_end_time, val_ppls, 'g', label='Validation')
    ax.set_xlabel('Wall-clock-time (minutes)')
    ax.set_ylabel('PPL')
    ax.legend(loc='upper right', shadow=True, fontsize='large')
    plt.show()


def problem_4_2():
    lc_path = "absolute path to use"

    train_ppls_1, val_ppls_1, epochs_end_time_1 = load_plot_values(lc_path)
    epochs_end_time_1 = [x/60.0 for x in epochs_end_time_1]
    epochs = range(len(train_ppls_1))

    # Learning curves over epochs
    fig, ax = plt.subplots()
    ax.plot(epochs, train_ppls_1, 'k', label='Training')
    ax.plot(epochs, val_ppls_1, 'g', label='Validation')
    # plt.xticks(np.arange(min(epochs), max(epochs) + 1, 1.0))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PPL')
    ax.legend(loc='upper right', shadow=True, fontsize='large')
    plt.show()

    # Learning curves over wall-clock-time
    fig, ax = plt.subplots()
    ax.plot(epochs_end_time_1, train_ppls_1, 'k', label='Training')
    ax.plot(epochs_end_time_1, val_ppls_1, 'g', label='Validation')
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
