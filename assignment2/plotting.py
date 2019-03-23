# We will use this script to do any plots we might need for the report

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def main():
    learning_curves = [
        "experiments_problem4_1/experiment_1_970gtx/learning_curves.npy",
        "experiments_problem4_1/experiment_2_970gtx/learning_curves.npy",
        "experiments_problem4_1/experiment_3_1070ti/learning_curves.npy",
        "experiments_problem4_2/experiment_4_970gtx/learning_curves.npy",
        "experiments_problem4_2/experiment_5_970gtx/learning_curves.npy",
        "experiments_problem4_2/experiment_6_1070ti/learning_curves.npy",
        "experiments_problem4_2/experiment_7_1070ti/learning_curves.npy",
        "experiments_problem4_2/experiment_8_1070ti/learning_curves.npy",
        "experiments_problem4_2/experiment_9_1070ti/learning_curves.npy",
        "experiments_problem4_3/experiment_10_970gtx_1/learning_curves.npy",
        "experiments_problem4_3/experiment_11_970gtx_1/learning_curves.npy",
        "experiments_problem4_3/experiment_12_970gtx_1/learning_curves.npy",
        "experiments_problem4_3/experiment_13_970gtx_1/learning_curves.npy",
        "experiments_problem4_3/experiment_14_970gtx_1/learning_curves.npy",
        "experiments_problem4_3/experiment_15_970gtx_1/learning_curves.npy",
        "experiments_problem4_3/experiment_16_1070ti_1/learning_curves.npy",
        "experiments_problem4_3/experiment_17_1070ti_1/learning_curves.npy",
        "experiments_problem4_3/experiment_18_1070ti_1/learning_curves.npy"
    ]

    draw_problem_4_1_figures(learning_curves)
    draw_problem_4_2_figures(learning_curves)
    draw_problem_4_3_figures(learning_curves)


def draw_problem_4_1_figures(learning_curves):
    figures_data = [
        {
            "title": "Experiment {}".format(i+1),
            "subplots": [
                {
                    "learning_curve": learning_curves[i],
                    "description": "",
                    "strokes": ["k", "g"],
                    "start_epoch": 0
                }
            ]
        }
        for i in range(3)
    ]
    figures_data[2]["subplots"][0]["start_epoch"] = 2

    for figure_data in figures_data:
        draw_epoch_and_wall_clock_time_figures(figure_data)


def draw_problem_4_2_figures(learning_curves):

    figures_data = [
        {
            "title": "Experiments 4 and 5: RNN architecture",
            "subplots": [
                {
                    "learning_curve": learning_curves[3],
                    "description": "#4",
                    "strokes": ["k", "g"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[4],
                    "description": "#5",
                    "strokes": ["k*-", "g*-"],
                    "start_epoch": 0
                }
            ]
        },
        {
            "title": "Experiments 6 and 7: GRU architecture",
            "subplots": [
                {
                    "learning_curve": learning_curves[5],
                    "description": "#6",
                    "strokes": ["k", "g"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[6],
                    "description": "#7",
                    "strokes": ["k*-", "g*-"],
                    "start_epoch": 0
                }
            ]
        },
        {
            "title": "Experiments 8 and 9: Transformer architecture",
            "subplots": [
                {
                    "learning_curve": learning_curves[7],
                    "description": "#8",
                    "strokes": ["k", "g"],
                    "start_epoch": 2
                },
                {
                    "learning_curve": learning_curves[8],
                    "description": "#9",
                    "strokes": ["k*-", "g*-"],
                    "start_epoch": 2
                }
            ]
        },
        {
            "title": "Experiments 4, 6 and 8: SGD Optimizer",
            "subplots": [
                {
                    "learning_curve": learning_curves[3],
                    "description": "#4 970gtx",
                    "strokes": ["k", "g"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[5],
                    "description": "#6 1070ti",
                    "strokes": ["b", "c"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[7],
                    "description": "#8 1070ti",
                    "strokes": ["k*-", "g*-"],
                    "start_epoch": 1
                }
            ]
        },
        {
            "title": "Experiment 5: SGD_LR_SCHEDULE optimizer",
            "subplots": [
                {
                    "learning_curve": learning_curves[4],
                    "description": "#5",
                    "strokes": ["k", "g"],
                    "start_epoch": 0
                },
            ]
        },
        {
            "title": "Experiments 7 and 9: ADAM Optimizer",
            "subplots": [
                {
                    "learning_curve": learning_curves[6],
                    "description": "#7",
                    "strokes": ["k", "g"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[8],
                    "description": "#9",
                    "strokes": ["k*-", "g*-"],
                    "start_epoch": 0
                }
            ]
        }
    ]

    for figure_data in figures_data:
        draw_epoch_and_wall_clock_time_figures(figure_data)


def draw_problem_4_3_figures(learning_curves):

    figures_data = [
        {
            "title": "Experiments 10, 11 and 12: RNN architecture",
            "subplots": [
                {
                    "learning_curve": learning_curves[9],
                    "description": "#10",
                    "strokes": ["k", "g"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[10],
                    "description": "#11",
                    "strokes": ["b", "c"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[11],
                    "description": "#12",
                    "strokes": ["k*-", "g*-"],
                    "start_epoch": 0
                }
            ]
        },
        {
            "title": "Experiments 13, 14 and 15: GRU architecture",
            "subplots": [
                {
                    "learning_curve": learning_curves[12],
                    "description": "#13",
                    "strokes": ["k", "g"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[13],
                    "description": "#14",
                    "strokes": ["b", "c"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[14],
                    "description": "#15",
                    "strokes": ["k*-", "g*-"],
                    "start_epoch": 0
                }
            ]
        },
        {
            "title": "Experiments 16, 17 and 18: Transformer architecture",
            "subplots": [
                {
                    "learning_curve": learning_curves[15],
                    "description": "#16",
                    "strokes": ["k", "g"],
                    "start_epoch": 2
                },
                {
                    "learning_curve": learning_curves[16],
                    "description": "#17",
                    "strokes": ["b", "c"],
                    "start_epoch": 2
                },
                {
                    "learning_curve": learning_curves[17],
                    "description": "#18",
                    "strokes": ["k*-", "g*-"],
                    "start_epoch": 0
                }
            ]
        },
        {
            "title": "Experiments 10, 13 and 16: SGD Optimizer",
            "subplots": [
                {
                    "learning_curve": learning_curves[9],
                    "description": "#10 970 GTX",
                    "strokes": ["k", "g"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[12],
                    "description": "#13 970 GTX",
                    "strokes": ["b", "c"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[15],
                    "description": "#16 1070 Ti",
                    "strokes": ["k*-", "g*-"],
                    "start_epoch": 2
                }
            ]
        },
        {
            "title": "Experiment 11, 14 and 17: SGD_LR_SCHEDULE optimizer",
            "subplots": [
                {
                    "learning_curve": learning_curves[10],
                    "description": "#11 970 GTX",
                    "strokes": ["k", "g"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[13],
                    "description": "#14 970 GTX",
                    "strokes": ["b", "c"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[16],
                    "description": "#17 1070 Ti",
                    "strokes": ["k*-", "g*-"],
                    "start_epoch": 3
                }
            ]
        },
        {
            "title": "Experiments 12, 15 and 18: ADAM Optimizer",
            "subplots": [
                {
                    "learning_curve": learning_curves[11],
                    "description": "#12 970 GTX",
                    "strokes": ["k", "g"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[14],
                    "description": "#15 970 GTX",
                    "strokes": ["b", "c"],
                    "start_epoch": 0
                },
                {
                    "learning_curve": learning_curves[17],
                    "description": "#18 1070 Ti",
                    "strokes": ["k*-", "g*-"],
                    "start_epoch": 0
                }
            ]
        }
    ]

    for figure_data in figures_data:
        draw_epoch_and_wall_clock_time_figures(figure_data)


def draw_epoch_and_wall_clock_time_figures(figure_data):
    # Learning curves over epochs
    ax = new_figure(figure_data["title"])
    for subplot in figure_data["subplots"]:
        train_ppls, val_ppls, epochs, epochs_end_time = load_data(subplot["learning_curve"], subplot["start_epoch"])

        ax.plot(epochs, train_ppls, subplot["strokes"][0], label=" ".join([subplot["description"], "Training"]))
        ax.plot(epochs, val_ppls, subplot["strokes"][1], label=" ".join([subplot["description"], "Validation"]))
    init_axis_and_legend(ax, 'Epoch', 'PPL')
    plt.show()

    # Learning curves over wall-clock-time
    ax = new_figure(figure_data["title"])
    for subplot in figure_data["subplots"]:
        train_ppls, val_ppls, epochs, epochs_end_time = load_data(subplot["learning_curve"], subplot["start_epoch"])

        ax.plot(epochs_end_time, train_ppls, subplot["strokes"][0], label=" ".join([subplot["description"], "Training"]))
        ax.plot(epochs_end_time, val_ppls, subplot["strokes"][1], label=" ".join([subplot["description"], "Validation"]))
    init_axis_and_legend(ax, 'Wall-clock-time (minutes)', 'PPL')
    plt.show()


def load_data(file_path, start_epoch):
    train_ppls, val_ppls, epochs_end_time = load_plot_values(file_path)
    train_ppls = train_ppls[start_epoch:]
    val_ppls = val_ppls[start_epoch:]
    epochs_end_time = [x / 60.0 for x in epochs_end_time[start_epoch:]]
    epochs = range(start_epoch, start_epoch + len(train_ppls))

    return train_ppls, val_ppls, epochs, epochs_end_time


def new_figure(figure_title):
    fig, ax = plt.subplots()
    ax.set_title(figure_title)
    return ax


def init_axis_and_legend(ax, x_axis_label, y_axis_label):
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.legend(loc='upper right', shadow=True, fontsize='large')


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
