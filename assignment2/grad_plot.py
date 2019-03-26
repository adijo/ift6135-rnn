import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')


def main():
    rnn_grad = np.load("grad_plot_data/rnn_grad.npy")
    gru_grad = np.load("grad_plot_data/gru_grad.npy")
    plt.plot(rnn_grad, label="rnn", linestyle="--", alpha=0.5, color="#001f3f")
    plt.plot(gru_grad, label="gru", linestyle="--", alpha=0.5, color="#FF4136")
    plt.legend()
    plt.xlabel("Time step (t)")
    plt.ylabel("Norm of grad of hidden layers wrt loss of last time step")
    plt.title("Vanishing Gradients")
    plt.show()


if __name__ == '__main__':
    main()
