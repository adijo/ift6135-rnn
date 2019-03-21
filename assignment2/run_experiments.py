import subprocess

original_4_1_and_4_2_commands = [
    "python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
    "python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35",
    "python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
    "python ptb-lm.py --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
    "python ptb-lm.py --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
    "python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
    "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9",
    "python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9",
    "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9"
]


def main():
    problem_4_3()


def problems_4_1_and_4_2():
    # The saved models (best parameters) can be found on
    # https://drive.google.com/drive/folders/1CeaePSAqsOERrAY6zxIqkVKJm75TyB1q?usp=sharing
    # as they were too big for git
    experiments = [
        "python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=experiments_problem4_1",
        "python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=experiments_problem4_1"
        "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best --save_dir=experiments_problem4_1"
    ]

    for command in experiments:
        for message in run_command(command):
            print(message, end="")


def problem_4_3():
    experiments = [
        # experiment_10_970gtx_1: learning rate 0.0004 instead of 0.0001
        # "python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=0.0004 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
        # TODO: experiment 11: learning rate 5 instead of 1
        "python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=5 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35",
        # TODO: experiment 11: learning rate 10 instead of 1
        # "python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35",
        # experiment_12_970gtx_1: dp_keep_prob = 0.5 instead of 0.35
        #"python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.5",
        # TODO: experiment 13: dp_keep_prob 0.25 instead of 0.35
        "python ptb-lm.py --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.25",
        # TODO: experiment 14: learning rate 15 instead of 10
        #"python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=15 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
        # TODO: experiment 15: hidden_size 1400 instead of 1500
        "python ptb-lm.py --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1400 --num_layers=2 --dp_keep_prob=0.35",
        # TODO: experiment 16: >>> todo what to tune <<<
        "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9",
        # TODO: experiment 17: dp_keep_prob 0.5 instead of 0.9
        "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.5",
        # TODO: experiment 18: >>> todo what to tune <<<
        "python ptb-lm.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9"
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
