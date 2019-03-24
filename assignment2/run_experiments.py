import subprocess


def main():
    # Since they were too big for git, the saved models (best parameters) for experiments 1, 2 and 3 can be found on
    # https://drive.google.com/drive/folders/1CeaePSAqsOERrAY6zxIqkVKJm75TyB1q?usp=sharing
    experiments = [
        # Experiment 1
        "python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best",
        # Experiment 2
        "python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best",
        # Experiment 3
        "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9",
        # Experiment 4
        "python ptb-lm.py --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
        # Experiment 5
        "python ptb-lm.py --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
        # Experiment 6
        "python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
        # Experiment 7
        "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9",
        # Experiment 8
        "python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.9",
        # Experiment 9
        "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9",
        # Experiment 10
        "python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=0.0004 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
        # Experiment 11
        "python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=5 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35",
        # Experiment 12
        "python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.5",
        # Experiment 13
        "python ptb-lm.py --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.25",
        # Experiment 14
        "python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=15 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
        # Experiment 15
        "python ptb-lm.py --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1400 --num_layers=2 --dp_keep_prob=0.35",
        # Experiment 16
        "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=256 --seq_len=35 --hidden_size=256 --num_layers=6 --dp_keep_prob=0.85",
        # Experiment 17
        "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=256 --seq_len=35 --hidden_size=256 --num_layers=6 --dp_keep_prob=0.85",
        # Experiment 18
        "python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=256 --seq_len=35 --hidden_size=128 --num_layers=2 --dp_keep_prob=0.9"
    ]

    for command in experiments:
        for message in run_command(command):
            print(message, end="")


def run_command(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    popen.wait()


if __name__ == '__main__':
    main()
