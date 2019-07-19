# NMT-Koean-To-English

작성중...
- 한영변역기 개발 튜토리얼
- https://github.com/jungyeul/korean-parallel-corpora 데이터셋 사용

```
pip install -r requirement.txt
```

---

## 1. Seq2Seq
- epoch : 500
- 구현 코드

```python
# -*- coding:utf-8 -*-
import random
import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, embedding, seq_len, rnn_dim, n_layer=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.embedding = embedding
        self.seq_len = seq_len
        self.rnn_dim = rnn_dim
        self.n_layers = n_layer

        embedding_dim = embedding.embedding_dim
        self.bach_norm = nn.BatchNorm1d(seq_len)
        self.rnn = nn.LSTM(embedding_dim, rnn_dim, n_layer, dropout=dropout, batch_first=True, bidirectional=True)
        # nn.LSTM
        # batch_first = False : (seq_len, batch_size, dims)
        # batch_first = True  : (batch_size, seq_len, dims)

    def forward(self, inputs, length):
        # input => (batch_size, time_step)

        embedded = self.embedding(inputs)               # => (batch_size, time_step, dimension)
        embedded = self.bach_norm(embedded)             # => (batch_size, time_step, dimension)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True)
        outputs, (hidden, cell) = self.rnn(packed)      # => (batch_size, time_step, rnn_dim)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)    # => (batch_size, seq len, rnn_dims)
        del outputs

        # bidirectional rnn - hidden/cell concat
        hidden = hidden[:1] + hidden[1:]
        cell = cell[:1] + cell[1:]

        return hidden, cell


class DecoderRNN(nn.Module):
    def __init__(self, embedding, rnn_dim, out_dim, n_layer=1, dropout=0):
        super(DecoderRNN, self).__init__()
        self.embedding = embedding
        self.rnn_dim = rnn_dim
        self.out_dim = out_dim
        self.n_layer = n_layer

        embedding_dim = embedding.embedding_dim
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, rnn_dim, n_layer, dropout=dropout, batch_first=True)
        # nn.LSTM
        # batch_first = False : (seq_len, batch_size, dims)
        # batch_first = True  : (batch_size, seq_len, dims)
        self.out = nn.Linear(rnn_dim, out_dim)

    def forward(self, inputs, last_hidden, last_cell):
        # inputs => (batch_size)
        # last_hidden => (n_layer, batch_size, rnn_dim)
        # last_cell => (n_layer, batch_size, rnn_dim)

        embedded = self.embedding(inputs)               # => (batch_size, dimension)
        embedded = self.batch_norm(embedded)            # => (batch_size, dimension)
        embedded = embedded.unsqueeze(1)                # => (batch_size, 1, dimension)

        output, (hidden, cell) = self.rnn(embedded, (last_hidden, last_cell))   # => (batch_size, 1, rnn_dim)
        output = output.squeeze(1)                      # => (batch_size, rnn_dim)
        predication = self.out(output)                  # => (batch_size, voc_size)
        return predication, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_input, src_length, trg_input, teacher_forcing_ratio=0.5):
        # src_input => (batch_size, seq_len)
        # trg_input => (batch_size, seq_len)
        # teacher_forcing_ratio is probability to use teacher forcing_rate
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        hidden, cell = self.encoder(src_input, src_length)

        batch_size = trg_input.shape[0]
        max_len = trg_input.shape[1]
        trg_vocab_size = self.decoder.out_dim

        outputs = torch.zeros(batch_size, max_len, trg_vocab_size)  # => (batch_size, seq_len, voc_size)
        input_ = trg_input[:, 0]                                    # => (batch_size)

        for t in range(1, max_len):
            predication, hidden, cell = self.decoder(input_, hidden, cell)
            outputs[:, t] = predication
            values, indices = predication.max(dim=1)
            del values
            # apply teacher forcing ratio
            input_ = trg_input[:, t] if random.random() < teacher_forcing_ratio else indices

        return outputs                                              # => (batch_size, seq_len, voc_size)

```

---

## 2. Seq2Seq with Attention

- epoch : 500
- 구현 코드

```
# -*- coding:utf-8 -*-
import random
import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, embedding, seq_len, rnn_dim, n_layer, dropout_rate=0):
        super().__init__()
        self.embedding = embedding
        self.seq_len = seq_len
        self.rnn_dim = rnn_dim
        self.n_layer = n_layer

        embedding_dim = embedding.embedding_dim
        self.batch_norm = nn.BatchNorm1d(seq_len)
        self.rnn = nn.LSTM(embedding_dim, rnn_dim, n_layer, dropout=dropout_rate, batch_first=True, bidirectional=True)

    def forward(self, inputs, length):
        embedded = self.embedding(inputs)
        embedded = self.batch_norm(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True)
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, outputs_length = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # outputs => (batch_size, seq_len, rnn_dims)
        del outputs_length

        # bidirectional rnn - output/hidden/cell concat
        outputs = outputs[:, :, :self.rnn_dim] + outputs[:, :, self.rnn_dim:]
        hidden = hidden[:1] + hidden[1:]
        cell = cell[:1] + cell[1:]

        return outputs, hidden, cell


class DecoderAttentionRNN(nn.Module):
    def __init__(self, attention, embedding, rnn_dim, out_dim, n_layer=1, dropout_rate=0):
        super().__init__()
        self.attention = attention
        self.embedding = embedding
        self.out_dim = out_dim

        embedding_dim = embedding.embedding_dim
        self.batch_norm = nn.BatchNorm1d(1)
        self.rnn = nn.LSTM(embedding_dim, rnn_dim, n_layer, dropout=dropout_rate, batch_first=True)
        self.linear = nn.Linear(rnn_dim * 2, out_dim)

    def forward(self, src_outputs, tar_input, last_hidden, last_cell):
        # src_outputs => (batch_size, seq_len, rnn_dim)
        # tar_input => (batch_size)
        embedded = self.embedding(tar_input)        # => (batch_size, embedding_size)
        embedded = embedded.unsqueeze(1)            # => (batch_size, 1, embedding_size)
        embedded = self.batch_norm(embedded)        # => (batch_size, 1, embedding_size)

        dec_output, (dec_hidden, dec_cell) = self.rnn(embedded, (last_hidden, last_cell))  # => (batch_size, 1, rnn_dim)

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

        input_ = tar_input[:, 0]

        for t in range(1, max_len):
            (predication, hidden, cell), attention_weight = self.decoder(enc_output, input_, hidden, cell)
            outputs[:, t] = predication
            values, indices = predication.max(dim=1)
            del values
            input_ = (tar_input[:, t] if random.random() < teacher_forcing_rate else indices)

        return outputs


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
```
---

## 3. Convolution Seq2Seq