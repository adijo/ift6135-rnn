import subprocess


def main():
    problems_4_1_and_4_2()


def problems_4_1_and_4_2():
    # These are done (both from problem 4.1)
    # These can be found on https://drive.google.com/drive/folders/1CeaePSAqsOERrAY6zxIqkVKJm75TyB1q?usp=sharing
    # (they were too big for git to handle)
    done_commands = [
        "python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=experiments_problem4_1",
        "python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best --save_dir=experiments_problem4_1"
        "python ptb-lm.py --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best --save_dir=experiments_problem4_1"
    ]

    commands_in_progress_by_helgi = [
        "python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_dir=experiments_problem4_2",
        "python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35 --save_dir=experiments_problem4_2"
    ]

    commands_in_progress_by_raphael = [
        "python ptb-lm.py --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_dir=experiments_problem4_2",
        "python ptb-lm.py  --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9 --save_dir=experiments_problem4_2",
        "python ptb-lm.py --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_dir=experiments_problem4_2",
        "python ptb-lm.py  --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9 --save_dir=experiments/problem4_2"
    ]

    # Leftover experiments for problems 4.1 and 4.2
    # All done or in progress!
    leftover_commands = []

    # Place the commands in this array you want to run (they will be run one after another, not in parallel)
    commands_to_run = [
          #"python ptb-lm.py --model=RNN --optimizer=SGD --initial_lr=0.0001 --batch_size=200 --seq_len=40 --hidden_size=1500 --num_layers=4 --dp_keep_prob=0.3",
          #"python ptb-lm.py --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=200 --seq_len=35 --hidden_size=2000 --num_layers=3 --dp_keep_prob=0.4",
          #"python ptb-lm.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=200 --seq_len=35 --hidden_size=1000 --num_layers=4 --dp_keep_prob=0.5",
          #"python ptb-lm.py --model=GRU --optimizer=SGD --initial_lr=0.0001 --batch_size=200 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.3",
        #"python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=200 --seq_len=35 --hidden_size=2000 --num_layers=2 --dp_keep_prob=0.35",
        #"python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=200 --seq_len=40 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35",
        "python ptb-lm.py --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=200 --seq_len=35 --hidden_size=1500 --num_layers=4 --dp_keep_prob=0.35",
          #"python ptb-lm.py --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=200 --seq_len=30 --hidden_size=1750 --num_layers=2 --dp_keep_prob=0.5"
    ]

    for command in commands_to_run:
        for message in run_command(command):
            print(message, end="")


def run_command(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()


if __name__ == '__main__':
    main()
