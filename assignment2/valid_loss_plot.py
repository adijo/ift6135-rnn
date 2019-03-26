import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')


def main():
    rnn_loss = np.load("time_seq_losses_data/rnn_loss.npy")
    gru_loss = np.load("time_seq_losses_data/gru_loss.npy")
    trans_loss = np.load("time_seq_losses_data/trans_loss.npy")
    plt.plot(rnn_loss, marker="o", label="rnn", linestyle="--", alpha=0.5, color="#001f3f")
    plt.plot(gru_loss, marker="o", label="gru", linestyle="--", alpha=0.5, color="#FF4136")
    plt.plot(trans_loss, marker="o", label="transformer", linestyle="--", alpha=0.5, color="#3D9970")
    plt.legend()
    plt.xlabel("Time step (t)")
    plt.ylabel("Avg valid loss per time step")
    plt.show()


if __name__ == '__main__':
    main()
