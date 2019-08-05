# -*- coding:utf-8 -*-
import random
import torch
import torch.nn as nn


def stack_rnn(embedding_size, rnn_dim, n_layers):
    modules = nn.ModuleList()
    for i in range(n_layers):
        if i == 0:
            rnn = nn.LSTM(embedding_size, rnn_dim, batch_first=True, bidirectional=True)
        else:
            rnn = nn.LSTM(rnn_dim, rnn_dim, batch_first=True, bidirectional=True)
        modules.append(rnn)
    return modules


class EncoderRNN(nn.Module):
    def __init__(self, embedding, seq_len, rnn_dim, n_layer, dropout_rate=0, use_residual=True):
        super().__init__()
        self.embedding = embedding
        self.seq_len = seq_len
        self.rnn_dim = rnn_dim
        self.n_layer = n_layer
        self.use_residual = use_residual

        self.embedding_dim = embedding.embedding_dim
        self.dropout = nn.Dropout(p=dropout_rate)
        self.rnn = stack_rnn(embedding.embedding_dim, rnn_dim, n_layer)

    def forward(self, inputs, length):
        embedded = self.embedding(inputs)
        next_hidden, next_cell = [], []
        x = embedded
        outputs = None

        for i, rnn in enumerate(self.rnn):
            x = self.dropout(x)     # dropout
            packed = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True)
            outputs, (hidden, cell) = rnn(packed)
            outputs, outputs_length = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            del outputs_length

            # append hidden & cell
            # hidden =>  (2, batch_size, rnn_dim) - 2 = because bidirectional
            # cell   =>  (2, batch_size, rnn_dim) - 2 = because bidirectional
            next_hidden.append(hidden)
            next_cell.append(cell)

            # concat output
            dims = outputs.shape[2]
            dims = int(dims / 2)
            outputs = outputs[:, :, :dims] + outputs[:, :, dims:]

            # residual connection
            if self.use_residual and i != 0:
                outputs = outputs + x
            x = outputs

        hidden = torch.stack(next_hidden)   # (n_layer, 2, batch_size, rnn_dim)
        cell = torch.stack(next_cell)       # (n_layer, 2, batch_size, rnn_dim)
        return outputs, hidden, cell


class DecoderAttentionRNN(nn.Module):
    def __init__(self, attention, embedding, rnn_dim, out_dim, n_layer=1, dropout_rate=0, use_residual=True):
        super().__init__()
        self.attention = attention
        self.embedding = embedding
        self.out_dim = out_dim

        self.embedding_dim = embedding.embedding_dim
        self.rnn = stack_rnn(self.embedding_dim, rnn_dim, n_layer)
        self.use_residual = use_residual
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(rnn_dim * 2, out_dim)
        self.test_rnn = nn.LSTM(self.embedding_dim, rnn_dim, batch_first=True, bidirectional=True)

    def forward(self, src_outputs, tar_input, last_hidden, last_cell):
        # src_outputs => (batch_size, seq_len, rnn_dim)
        # tar_input => (batch_size)
        embedded = self.embedding(tar_input)        # => (batch_size, embedding_size)
        embedded = embedded.unsqueeze(1)            # => (batch_size, 1, embedding_size)

        next_hidden, next_cell = [], []
        dec_output = None
        x = embedded
        for i, rnn in enumerate(self.rnn):
            x = self.dropout(x)
            dec_output, (dec_hidden, dec_cell) = rnn(x, (last_hidden[i], last_cell[i]))
            # append hidden & cell
            next_hidden.append(dec_hidden)
            next_cell.append(dec_cell)

            # concat output
            dims = dec_output.shape[2]
            dims = int(dims / 2)
            dec_output = dec_output[:, :, :dims] + dec_output[:, :, dims:]

            # residual connection
            if self.use_residual and i != 0:
                dec_output = dec_output + x
            x = dec_output

        dec_hidden = torch.stack(next_hidden)   # (n_layer, 2, batch_size, rnn_dim)
        dec_cell = torch.stack(next_cell)       # (n_layer, 2, batch_size, rnn_dim)

        # calc attention distribution
        attention_distribution = self.attention(src_outputs, dec_output)    # => (batch_size, seq_len, 1)

        # calc attention value (= context vector)
        temp = src_outputs * attention_distribution                         # => (batch_size, seq_len, rnn_dim)
        context_vector = temp.sum(dim=1)                                    # => (batch_size, rnn_dim)

        # concat context vector
        dec_output = dec_output.squeeze(dim=1)                              # => (batch_size, rnn_dim)
        concat = torch.cat((dec_output, context_vector), dim=1)             # => (batch_size, rnn_dim * 2)
        predication = self.linear(concat)                                   # => (batch_size, out_dim)
        return (predication, dec_hidden, dec_cell), attention_distribution


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_input, src_length, tar_input, teacher_forcing_rate=0.5):
        enc_output, hidden, cell = self.encoder(src_input, src_length)

        batch_size, max_len = tar_input.shape
        out_dim = self.decoder.out_dim
        outputs = torch.zeros((batch_size, max_len, out_dim))
        attention_weight_list = []

        input_ = tar_input[:, 0]

        for t in range(1, max_len):
            (predication, hidden, cell), attention_weight = self.decoder(enc_output, input_, hidden, cell)
            # attention_weight => (batch_size, seq_len, 1)
            attention_weight = attention_weight.squeeze(dim=2)
            attention_weight_list.append(attention_weight)
            outputs[:, t] = predication
            values, indices = predication.max(dim=1)
            del values
            input_ = (tar_input[:, t] if random.random() < teacher_forcing_rate else indices)

        attention_weights = torch.stack(attention_weight_list, dim=2)   # attention
        return outputs, attention_weights


class Attention(nn.Module):
    """
    for which we consider three different alternatives:

    Score function

    score(ht, hs) =
        htT * hS                =>  (dot)
        htT * W * hs            =>  (general)
        vT * tanh * (W*[ht;hs]) =>  (concat)
    """

    def __init__(self, method, rnn_dim=None):
        super().__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise NotImplementedError('implement [dot, general, concat]')

        if self.method == 'general':
            self.nn = nn.Linear(rnn_dim, rnn_dim)
        if self.method == 'concat':
            self.nn = nn.Linear(rnn_dim * 2, rnn_dim)
            self.v = nn.Linear(rnn_dim, 1)

    def dot(self, src_inputs, tar_input):
        """
            src_inputs => (batch_size, seq_len, rnn_dim)
            tar_input => (batch_size, rnn_dim, 1)

            < dot product >

            => src_input * tar_input
            => (batch_size, seq_len, rnn_dim) * (batch_size, rnn_dim, 1) = (batch_size, seq_len, 1)
        """
        attention_value = src_inputs.bmm(tar_input)      # (batch_size, seq_len, 1)
        return attention_value

    def general(self, src_inputs, tar_input):
        """
            src_inputs => (batch_size, seq_len, rnn_dim)
            tar_input => (batch_size, rnn_dim, 1)
            weight => (rnn_dim, rnn_dim)

            < general >

            => src_input x weight x tar_input
            => (batch_size, seq_len, rnn_dim) x (rnn_dim, rnn_dim) x (batch_size, rnn_dim, 1) = (batch_size, seq_len, 1)
        """
        attention_value = self.nn(src_inputs)
        attention_value = attention_value.bmm(tar_input)    # => (batch_size, seq_len, 1)
        return attention_value

    def concat(self, src_inputs, tar_input):
        """
            src_inputs => (batch_size, seq_len, rnn_dim)
            tar_input => (batch_size, rnn_dim, 1)
        """
        src_inputs = src_inputs.permute(0, 2, 1)                    # => (batch_size, rnn_dim, seq_len)
        tar_input = tar_input.expand(-1, -1, src_inputs.size(2))    # => (batch_size, rnn_dim, seq_len)
        hidden = torch.cat((src_inputs, tar_input), dim=1)          # => (batch_size, rnn_dim * 2, seq_len)

        # 1. hidden
        #       => (batch_size, rnn_dim * 2, seq_len)
        #
        # 2. hidden permute
        #       => (batch_size, seq_len, rnn_dim * 2)
        #
        # 3. hidden x weight
        #       => (batch_size, seq_len, rnn_dim * 2) * (rnn_dim * 2, rnn_dim) = (batch_size, seq_len, rnn_dim)
        #
        # 4. tanh
        #       => (batch_size, seq_len, rnn_dim)
        #
        # 5. v * hidden
        #       => (batch_size, seq_len, rnn_dim) * (rnn_dim, 1) = (batch_size, seq_len, 1)

        attention_value = self.v(torch.tan(self.nn(hidden.permute(0, 2, 1))))
        return attention_value

    def forward(self, src_inputs, tar_input):
        # src_input => (batch_size, seq_len, rnn_dim)
        # tar_input => (batch_size, 1, rnn_dim)

        # target transpose
        tar_input = tar_input.permute(0, 2, 1)                          # => (batch_size, rnn_dim, 1)

        attention_value = None
        if self.method == 'dot':
            attention_value = self.dot(src_inputs, tar_input)
        elif self.method == 'general':
            attention_value = self.general(src_inputs, tar_input)
        elif self.method == 'concat':
            attention_value = self.concat(src_inputs, tar_input)

        # attention_value => (batch_size, seq_len, 1)
        attention_distribution = nn.Softmax(dim=1)(attention_value)     # => (batch_size, seq_len, 1)
        return attention_distribution
