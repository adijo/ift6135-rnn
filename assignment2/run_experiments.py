import subprocess


def main():
    problem_4_3()


def problems_4_1_and_4_2():
    # These are done (both from problem 4.1)
    # These can be found on https://drive.google.com/drive/folders/1CeaePSAqsOERrAY6zxIqkVKJm75TyB1q?usp=sharing
    # (they were too big for git to handle)
    done_commands = [
        "python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=experiments_problem4_1",
        "python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=experiments_problem4_1"
        "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best --save_dir=experiments_problem4_1"
    ]


def problem_4_3():
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

    attempted_improved_4_1_and_4_2_commands_to_use_for_4_3 = [
        # learning rate 0.0004 instead of 0.0001
        "python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=0.0004 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
        # learning rate 5 instead of 1
        "python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=5 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35",
        # learning rate 10 instead of 1
        #"python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35",
        # learning rate 15 instead of 10
        #"python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=15 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
        # dp_keep_prob 0.5 instead of 0.9
        "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.5",
        # dp_keep_prob 0.25 instead of 0.35
        "python ptb-lm.py --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.25",
        # hidden_size 1400 instead of 1500
        "python ptb-lm.py --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1400 --num_layers=2 --dp_keep_prob=0.35"
    ]

    for command in attempted_improved_4_1_and_4_2_commands_to_use_for_4_3:
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
