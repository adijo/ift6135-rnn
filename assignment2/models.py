import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Problem 1
# Implement a stacked vanilla RNN with Tanh nonlinearities.
class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()

        # TODO ========================
        # Initialization of the parameters of the recurrent and fc layers.
        # Your implementation should support any number of stacked hidden layers
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
        # modules, but not recurrent modules.
        #
        # To create a variable number of parameter tensors and/or nn.Modules
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the
        # provided clones function (as opposed to a regular python list), in order
        # for Pytorch to recognize these parameters as belonging to this nn.Module
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.p = 1 - dp_keep_prob

        self.drop = nn.Dropout(self.p)

        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.FirstHidden_input = nn.Linear(emb_size, hidden_size)
        self.FirstHidden_hidden = nn.Linear(hidden_size, hidden_size)

        sublayer = nn.ModuleList([nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, hidden_size)])
        self.hidden_layers = clones(sublayer, self.num_layers - 1)

        self.Wy = nn.Linear(hidden_size, vocab_size)

        self.apply(self.init_weights_uniform)

    def init_weights_uniform(self, m):
        # TODO ========================
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
        # in the range [-k, k] where k is the square root of 1/hidden_size
        initrange = 0.1
        if type(m) == nn.Linear:
            torch.nn.init.uniform_(m.weight, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
        if type(m) == nn.Linear:
            torch.nn.init.uniform_(m.bias, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
        if type(m) == nn.Embedding:
            torch.nn.init.uniform_(m.weight, -initrange, initrange)
        self.Wy.bias.data.zero_()
        self.Wy.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self):
        # TODO ========================
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        # if we do what is asked here the initial hidden state will not be learned but rather fixed at 0.
        self.inititial_hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, requires_grad=True)

        # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
        return self.inititial_hidden

    def forward(self, inputs, hidden):
        # TODO ========================
        # Compute the forward pass, using nested python for loops.
        # The outer for loop should iterate over timesteps, and the
        # inner for loop should iterate over hidden layers of the stack.
        #
        # Within these for loops, use the parameter tensors and/or nn.modules you
        # created in __init__ to compute the recurrent updates according to the
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details,
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """
        hiddendict = {}
        logits_list = []

        hiddendict[0] = hidden

        for timestep in range(inputs.size(0)):
            hiddendict[timestep + 1] = {}

            embedded = self.drop(self.embedding(inputs[timestep]))
            hiddendict[timestep + 1][0] = torch.tanh(self.FirstHidden_input(embedded) +
                                                     self.FirstHidden_hidden(hiddendict[timestep][0]))

            for i, layer in enumerate(self.hidden_layers):
                x = self.drop(hiddendict[timestep + 1][i])
                hiddendict[timestep + 1][i + 1] = torch.tanh(layer[0](x) + layer[1](hiddendict[timestep][i + 1]))

            logits_list.append(self.Wy(self.drop(hiddendict[timestep + 1][self.num_layers - 1])).unsqueeze(0))

        logits = torch.cat(logits_list, dim=0)
        hidden = torch.cat([hiddendict[self.seq_len][h].unsqueeze(0) for h in range(self.num_layers)], dim=0)

        return logits.view(inputs.size(0), inputs.size(1), self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """

        return samples


# Problem 2
# Implement a stacked GRU RNN
class GRU(nn.Module):
    """
    Follow the same instructions as for RNN (above), but use the equations for
    GRU, not Vanilla RNN.
    """

    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()

        # TODO ========================
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.p = 1 - dp_keep_prob
        self.drop = nn.Dropout(self.p)

        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.First_Wr = nn.Linear(emb_size, hidden_size)
        self.First_Ur = nn.Linear(hidden_size, hidden_size)

        self.First_Wz = nn.Linear(emb_size, hidden_size)
        self.First_Uz = nn.Linear(hidden_size, hidden_size)

        self.First_Wh = nn.Linear(emb_size, hidden_size)
        self.First_Uh = nn.Linear(hidden_size, hidden_size)

        sublayer = nn.ModuleList([nn.Linear(hidden_size, hidden_size),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.Linear(hidden_size, hidden_size)])

        self.hidden_layers = clones(sublayer, self.num_layers - 1)

        self.Wy = nn.Linear(hidden_size, vocab_size)

        self.apply(self.init_weights_uniform)

    def init_weights_uniform(self, m):
        # TODO ========================
        # Initialize all the weights uniformly in the range [-0.1, 0.1]
        # and all the biases to 0 (in place)
        # if this is done performance is quite bad
        # Weight for emmbedding and last layer initialize all the weights uniformly in the range [-0.1, 0.1]
        # Weight for the reccurent part  uniformly in the range [-sqrt(1/nb hidden_unit), sqrt(1/nb hidden_unit)]
        initrange = 0.1
        if type(m) == nn.Linear:
            torch.nn.init.uniform_(m.weight, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
        if type(m) == nn.Linear:
            torch.nn.init.uniform_(m.bias, -math.sqrt(1 / self.hidden_size), math.sqrt(1 / self.hidden_size))
        if type(m) == nn.Embedding:
            torch.nn.init.uniform_(m.weight, -initrange, initrange)
        self.Wy.bias.data.zero_()
        self.Wy.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self):
        # TODO ========================
        self.inititial_hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        return self.inititial_hidden

    def forward(self, inputs, hidden):
        # TODO ========================

        logits_list = []
        hiddendict = {}
        logits_list = []
        hiddendict[0] = hidden

        for timestep in range(inputs.size(0)):
            hiddendict[timestep + 1] = {}

            embedded = self.drop(self.embedding(inputs[timestep]))
            reset = torch.sigmoid(self.First_Wr(embedded) + self.First_Ur(hiddendict[timestep][0]))
            forget = torch.sigmoid(self.First_Wz(embedded) + self.First_Uz(hiddendict[timestep][0]))
            hidden_mod = torch.tanh(self.First_Wh(embedded) + self.First_Uh(reset * hiddendict[timestep][0]))
            hiddendict[timestep + 1][0] = (1 - forget) * hidden_mod + forget * hiddendict[timestep][0]

            for i, layer in enumerate(self.hidden_layers):
                x = self.drop(hiddendict[timestep + 1][i])  # apply Dropout
                reset = torch.sigmoid(layer[0](x) + layer[1](hiddendict[timestep][i + 1]))
                forget = torch.sigmoid(layer[2](x) + layer[3](hiddendict[timestep][i + 1]))
                hidden_mod = torch.tanh(layer[4](x) + layer[5](reset * hiddendict[timestep][i + 1]))
                hiddendict[timestep + 1][i + 1] = forget * hidden_mod + (1 - forget) * hiddendict[timestep][i + 1]

            logits_list.append(self.Wy(self.drop(hiddendict[timestep + 1][self.num_layers - 1])).unsqueeze(0))

        logits = torch.cat(logits_list, dim=0)
        hidden = torch.cat([hiddendict[self.seq_len][h].unsqueeze(0) for h in range(self.num_layers)], dim=0)

        return logits.view(inputs.size(0), inputs.size(1), self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================

        return samples


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""


# This code has been modified from an open-source project, by David Krueger.
# The original license is included below:
# MIT License
#
# Copyright (c) 2018 Alexander Rush
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ----------------------------------------------------------------------------------


# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units 
        self.n_heads = n_heads
        self.dropout_rate = dropout

        self.dropout = nn.Dropout(p=dropout)

		# TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
		
        self.V = nn.ModuleList([nn.Linear(n_units,self.d_k) for i in range(n_heads)])
        self.Q = nn.ModuleList([nn.Linear(n_units,self.d_k) for i in range(n_heads)])
        self.K = nn.ModuleList([nn.Linear(n_units,self.d_k) for i in range(n_heads)])

        self.O = nn.Linear(n_units,n_units)

        self.apply(self.init_weights)
             
    def init_weights(self, m):
        # TODO ========================
         # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        if type(m) == nn.Linear:
            torch.nn.init.uniform_(m.weight, -math.sqrt(1/m.out_features), math.sqrt(1/m.out_features))
            torch.nn.init.uniform_(m.bias, -math.sqrt(1/m.out_features), math.sqrt(1/m.out_features))

        
    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value all have size: (batch_size, seq_len, self.n_units, self.d_k)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softt)
        # Also apply dropout to the attention values.
        self.v = []
        self.q = []
        self.k = []
        self.complete_attention = torch.empty((0,0)).cuda()
        for i in range(0,self.n_heads):
            self.v.append(self.V[i](query)) #Q*W_v[i] + b_v
            self.q.append(self.Q[i](query)) #Q*W_q[i] + b_q
            self.k.append(self.K[i](query)) #Q*W_k[i] + b_k
            self.complete_attention = torch.cat((self.complete_attention, self.attention(self.q[i],self.k[i],self.v[i], mask)), dim=2)

        self.o = self.O(self.complete_attention) # Attention = concat(Attention[1] ... Attention[n]*W_o + b_o)
        return self.o# size: (batch_size, seq_len, self.n_units)

    def stable_softmax(self, x, s): #X is the input vector, s is the dropout mask
        s = s.float() #128 x 35 x 35

        #This implentation had crazy validation / train ppl. epoch: 21    train ppl: 1.1572353641563802    val ppl: 1.4495772582466449. 
        #It acts as if there was not mask at all, and the predictions are made using all the 35 words instead of only the previous words.
        #x_tilde = x * s - 1e-9*(1-s)
        
        #Implementation of the mask, as specified in the paper:
        #x_tilde = x.masked_fill(s==0, -float("inf")) #Paper is clear on the fact the the masked value should be -inf before applying the softmax https://arxiv.org/pdf/1706.03762.pdf
        
        #Appently I just did a typo. But the formula x * s is clearly not equivalent to this one.
        x_tilde = x * s - 1e9*(1-s)
        
        #According to the latest post in slack, we should use torch.softmax.
        result = torch.nn.functional.softmax(x_tilde, dim=2)
        #exp_x_tilde = torch.exp(x_tilde)
        #sum_x_tilde = torch.sum(exp_x_tilde,dim=2)
        #sum_x_tilde = sum_x_tilde.unsqueeze(dim=2)
        #result = exp_x_tilde / sum_x_tilde
        return result

    def attention(self, Q, K, V, s):
        K_t = K.transpose(1,2)
        qkt_on_dk = torch.matmul(Q, K_t)/math.sqrt(self.d_k)
        qkt_on_dk_ss = self.stable_softmax(qkt_on_dk,s)
        qkt_on_dk_ss = self.dropout(qkt_on_dk_ss)
        res = torch.matmul(qkt_on_dk_ss,V)
        return res


# ----------------------------------------------------------------------------------
# The encodings of elements of the input sequence


class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# ----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer

class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # apply the self-attention
        return self.sublayer[1](x, self.feed_forward)  # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """

    def __init__(self, layer, n_blocks):  # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, n_units=512, n_heads=16, dropout=0.1):
    """
    Helper: Construct a model from hyper-parameters.
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# ----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    """
    layer normalization, as in: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """

    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
