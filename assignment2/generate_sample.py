from models import RNN, GRU
import torch
import argparse
import os
import collections

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus')
parser.add_argument('--GRU_path', type=str, default='gru/best_params.pt',
                    help='path to the GRU model')
parser.add_argument('--RNN_path', type=str, default='rnn/best_params.pt',
                    help='path to the RNN model')
parser.add_argument('--seq_len', type=int, default=35,
                    help='Sequence length used to train the models')
parser.add_argument('--batch_size', type=int, default=10,
                    help='size of one minibatch')
parser.add_argument('--GRU_hidden_size', type=int, default=1500,
                    help='size of hidden layers for the GRU')
parser.add_argument('--RNN_hidden_size', type=int, default=1500,
                    help='size of hidden layers for the RNN')
parser.add_argument('--GRU_num_layers', type=int, default=2,
                    help='number of hidden layers in GRU')
parser.add_argument('--RNN_num_layers', type=int, default=2,
                    help='number of hidden layers in RNN')
parser.add_argument('--GRU_emb_size', type=int, default=200,
                    help='size of word embeddings in GRU')
parser.add_argument('--RNN_emb_size', type=int, default=200,
                    help='size of word embeddings in RNN')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

# Build the dictionnary
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def print_sentence(sentence):
    for line in range(sentence.size()[1]):
        print(" ".join([id_2_word[id.item()] for id in sentence[:,line]]))

args = parser.parse_args()
argsdict = args.__dict__

train_path = os.path.join(args.data, "ptb" + ".train.txt")
word_to_id, id_2_word = _build_vocab(train_path)
vocab_size = len(word_to_id)

# Create the model
rnn = RNN(emb_size=argsdict["RNN_emb_size"], hidden_size=argsdict["RNN_hidden_size"],
            seq_len=argsdict["seq_len"], batch_size=argsdict["batch_size"],
            vocab_size=vocab_size, num_layers=argsdict["RNN_num_layers"],
            dp_keep_prob=1)

gru = GRU(emb_size=argsdict["GRU_emb_size"], hidden_size=argsdict["GRU_hidden_size"],
                seq_len=argsdict["seq_len"], batch_size=argsdict["batch_size"],
                vocab_size=vocab_size, num_layers=argsdict["GRU_num_layers"],
                dp_keep_prob=1)

# Load the model weight
rnn.load_state_dict(torch.load("rnn/best_params.pt"))
gru.load_state_dict(torch.load("gru/best_params.pt"))

rnn.eval()
gru.eval()

# Initialize the hidden state
hidden = [rnn.init_hidden(), gru.init_hidden()]

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Generate the word seed using random words
# in the first 100 most common words.
input = torch.randint(0, 100, (args.batch_size, 1)).squeeze()


for name_model, model, init_hidden in zip(["RNN", "GRU"], [rnn, gru], hidden):
    print("------------------------------------")
    print(name_model)
    print("------------------------------------")
    print_sentence(model.generate(input, init_hidden, args.seq_len))
    print_sentence(model.generate(input, init_hidden, 2*args.seq_len))

