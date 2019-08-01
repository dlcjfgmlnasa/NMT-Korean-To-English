# -*- coding:utf-8 -*-
"""
This code reference https://github.com/bentrevett/pytorch-seq2seq
Convolution Sequence to Sequence code
"""

import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PositionEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dims):
        super().__init__()
        self.num_embeddings = num_embeddings    # vocab size
        self.embedding_dims = embedding_dims    # embedding dim size

        self.word_embedding = nn.Embedding(num_embeddings, embedding_dims)
        self.pos_embedding = nn.Embedding(100, embedding_dims)

    def forward(self, inputs):
        pos = torch.arange(0, inputs.shape[1]).repeat(inputs.shape[0], 1).to(device)
        embedded = self.word_embedding(inputs) + self.pos_embedding(pos)
        return embedded


class GLU(nn.Module):
    # gated linear unit (Dauphin etal, 2016)
    def __init__(self):
        super().__init__()

    def forward(self, inputs, dim):
        split_point = int(inputs.shape[dim] / 2)
        a = inputs[:, :split_point, :]
        b = inputs[:, split_point:, :]
        result = a * torch.sigmoid(b)
        return result


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dims, n_layers, hidden_size, kernel_size):
        super().__init__()
        self.num_embeddings = num_embeddings    # vocab size
        self.embedding_dims = embedding_dims
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.embedding = PositionEmbedding(num_embeddings, embedding_dims)
        self.emd2hid = nn.Linear(embedding_dims, hidden_size)
        self.hid2emd = nn.Linear(hidden_size, embedding_dims)

        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size*2,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2
            ) for _ in range(n_layers)
        ])
        self.glu = GLU()

    def forward(self, src_input):
        # src_input => (batch_size, seq_len)
        embedded = self.embedding(src_input)                # => (batch_size, seq_len, embedding_size)
        conv_input = self.emd2hid(embedded)                 # => (batch_size, seq_len, hidden_size)
        conv_input = conv_input.permute(0, 2, 1)            # => (batch_size, hidden_size, seq_len)

        # cnn network
        for convolution in self.convolutions:
            result = convolution(conv_input)                # => (batch_size, hidden_size * 2, seq_len)
            result = self.glu(result, dim=1)                # => (batch_size, hidden_size, seq_len)

            # residual connection
            conv_input = (conv_input + result) * self.scale     # => (batch_size, hidden_size, seq_len)

        conv_input = conv_input.permute(0, 2, 1)            # => (batch_size, seq_len, hidden_size)
        conved = self.hid2emd(conv_input)                   # => (batch_size, seq_len, embedding_size)

        # elementwise sum conved and embedded to be used for attention
        combined = (conved + embedded) * self.scale         # => (batch_size, seq_len, embedding_size)
        return conved, combined


class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dims, n_layers, hidden_size, kernel_size, pad_id):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.pad_id = pad_id

        self.embedding = PositionEmbedding(num_embeddings, embedding_dims)

        self.emb2hid = nn.Linear(embedding_dims, hidden_size)
        self.hid2emd = nn.Linear(hidden_size, embedding_dims)
        self.att_emb2hid = nn.Linear(embedding_dims, hidden_size)
        self.att_hid2emd = nn.Linear(hidden_size, embedding_dims)
        self.out = nn.Linear(embedding_dims, num_embeddings)

        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size*2,
                kernel_size=kernel_size
            )
            for _ in range(n_layers)
        ])
        self.glu = GLU()

    def attention(self, embedded, dec_conved, enc_conved, enc_combined):
        # embedded     => (batch_size, seq_len, embedding_size)
        # dec_conved   => (batch_size, hidden_size, seq_len)
        # enc_conved   => (batch_size, seq_len, embedding_size)
        # enc_combined => (batch_size, seq_len, embedding_size)

        dec_conv_emd = self.att_hid2emd(dec_conved.permute(0, 2, 1))  # => (batch_size, seq_len, embedding_size)
        dec_combined = (embedded + dec_conv_emd) * self.scale         # => (batch_size, seq_len, embedding_size)

        # attention_matrix
        #  = (batch_size, seq_len, embedding_size) * (batch_size, embedding_size, seq_len)
        #  = (batch_size, seq_len, seq_len)
        attention_matrix = torch.matmul(dec_combined, enc_combined.permute(0, 2, 1))
        attention_score = torch.softmax(attention_matrix, dim=2)    # => (batch_size, seq_len, seq_len)

        # conditional_input
        #  = (batch_size, seq_len, seq_len) *
        #           {(batch_size, seq_len, embedding_size) + (batch_size, seq_len, embedding_size)}
        #  = (batch_size, seq_len, embedding_size)
        conditional_input = torch.matmul(attention_score, (enc_conved + enc_combined))
        conditional_input = self.att_emb2hid(conditional_input)     # => (batch_size, seq_len, hidden_size)
        attention_combined = (dec_conved + conditional_input.permute(0, 2, 1)) * self.scale
        return attention_score, attention_combined

    def forward(self, trg_input, enc_conved, enc_combined):
        embedded = self.embedding(trg_input)                # => (batch_size, seq_len, embedding_size)
        conv_input = self.emb2hid(embedded)                 # => (batch_size, seq_len, hidden_size)
        conv_input = conv_input.permute(0, 2, 1)            # => (batch_size, hidden_size, seq_len)

        conved = None
        attention_score = None
        for convolution in self.convolutions:
            padding = torch.zeros(conv_input.shape[0], conv_input.shape[1], self.kernel_size-1).fill_(self.pad_id).\
                to(device)
            padded_conv_input = torch.cat((padding, conv_input), dim=2)     # => (batch_size, hidden_size, seq_len+2)

            # result => (batch_size, hidden_size * 2, seq_len + kernel_size - 1)
            conved = convolution(padded_conv_input)
            conved = self.glu(conved, dim=1)                                # => (batch_size, hidden_size, seq_len)
            attention_score, conved = self.attention(embedded, conved, enc_conved, enc_combined)

            # residual connection
            conved = (conved + conv_input) * self.scale                     # => (batch_size, hidden_size, seq_len)
            conv_input = conved
        output = self.hid2emd(conved.permute(0, 2, 1))                      # => (batch_size, seq_len, hidden_size)
        output = self.out(output)                                           # => (batch_size, seq_len, output_dim)
        return output, attention_score


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_input, trg_input, trg_output):
        enc_convolution, enc_combined = self.encoder(src_input)
        output, attention_score = self.decoder(trg_input, enc_convolution, enc_combined)
        return output, attention_score
