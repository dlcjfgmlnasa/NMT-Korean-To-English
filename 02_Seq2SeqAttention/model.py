# -*- coding:utf-8 -*-
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class StackRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bias=True, dropout=0, residual=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.residual = residual

        for i in range(n_layers):
            self.layers.append(
                nn.LSTMCell(input_size, hidden_size, bias=bias)
            )
            input_size = hidden_size

    def forward(self, inputs, hidden):
        h_state, c_state = hidden

        next_h_state, next_c_state = [], []

        for i, layer in enumerate(self.layers):
            hi = h_state[i].squeeze(dim=0)
            ci = c_state[i].squeeze(dim=0)

            if hi.dim() == 1 and ci.dim() == 1:
                hi = h_state[i]
                ci = c_state[i]

            next_hi, next_ci = layer(inputs, (hi, ci))
            output = next_hi

            if i + 1 < self.n_layers:
                # rnn dropout layer
                output = self.dropout(output)
            if self.residual and inputs.size(-1) == output.size(-1):
                # residual connection
                inputs = output + inputs
            else:
                inputs = output
            next_h_state.append(next_hi)
            next_c_state.append(next_ci)

        next_hidden = (
            torch.stack(next_h_state, dim=0),
            torch.stack(next_c_state, dim=0)
        )
        return inputs, next_hidden


class StackAttentionCell(StackRNNCell):
    def __init__(self, input_size, hidden_size, attention, n_layers=1, bias=True, dropout=0, residual=True):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            bias=bias,
            dropout=dropout,
            residual=residual
        )
        self.attention = attention

    def forward(self, inputs_with_context, hidden, get_attention=False):
        inputs, context = inputs_with_context
        h_state, c_state = hidden

        next_h_state, next_c_state = [], []
        for i, layer in enumerate(self.layers):
            hi = h_state[i].squeeze(dim=0)
            ci = c_state[i].squeeze(dim=0)

            if hi.dim() == 1 and ci.dim() == 1:
                hi = h_state[i]
                ci = c_state[i]

            next_hi, next_ci = layer(inputs, (hi, ci))
            output = next_hi

            if i + 1 < self.n_layers:
                # rnn dropout layer
                output = self.dropout(output)
            if self.residual and inputs.size(-1) == output.size(-1):
                # residual connection
                inputs = output + inputs
            else:
                inputs = output
            next_h_state.append(next_hi)
            next_c_state.append(next_ci)

        next_hidden = (
            torch.stack(next_h_state, dim=0),
            torch.stack(next_c_state, dim=0)
        )
        attention_distribution, context_vector = self.attention(context, inputs)
        if get_attention:
            return context_vector, attention_distribution, next_hidden
        else:
            del attention_distribution
            return context_vector, next_hidden


class Recurrent(nn.Module):
    def __init__(self, cell, with_attention=False, reverse=False):
        super().__init__()
        self.cell = cell
        self.with_attention = with_attention
        self.reverse = reverse

    def forward(self, inputs, hidden=None, context=None, get_attention=False):
        hidden_size = self.cell.hidden_size
        batch_size = inputs.size()[0]

        if hidden is None:
            n_layers = self.cell.n_layers
            zero = inputs.data.new(1).zero_()
            h0 = zero.view(1, 1, 1).expand(n_layers, batch_size, hidden_size)
            c0 = zero.view(1, 1, 1).expand(n_layers, batch_size, hidden_size)
            hidden = (h0, c0)

        outputs = []
        attentions = []
        inputs_time = inputs.split(1, dim=1)
        if self.reverse:
            inputs_time = list(inputs_time)
            inputs_time.reverse()

        for input_t in inputs_time:
            input_t = input_t.squeeze(1)
            if self.with_attention:
                input_t = (input_t, context)
                if get_attention:
                    output_t, score, hidden = self.cell(input_t, hidden, get_attention)
                    attentions.append(score.squeeze(dim=2))
                else:
                    output_t, hidden = self.cell(input_t, hidden, get_attention)
            else:
                output_t, hidden = self.cell(input_t, hidden)
            outputs += [output_t]

        if self.reverse:
            outputs.reverse()

        outputs = torch.stack(outputs, dim=1)
        if get_attention:
            attentions = torch.stack(attentions, dim=2)
            return outputs, attentions, hidden
        return outputs, hidden


class BiRecurrent(nn.Module):
    def __init__(self, cell, output_transformer, output_transformer_bias,
                 hidden_transformer, hidden_transformer_bias):
        super().__init__()
        hidden_size = cell.hidden_size * 2
        self.forward_rnn = Recurrent(cell, reverse=False)
        self.reverse_rnn = Recurrent(cell, reverse=True)
        self.output_nn = nn.Linear(hidden_size, output_transformer, bias=output_transformer_bias)
        self.hidden_nn = nn.Linear(hidden_size, hidden_transformer, bias=hidden_transformer_bias)
        self.cell_nn = nn.Linear(hidden_size, hidden_transformer, bias=hidden_transformer_bias)

    def forward(self, inputs, hidden=None):
        forward_output, (forward_hidden, forward_cell) = self.forward_rnn(inputs, hidden)
        reverse_output, (reverse_hidden, reverse_cell) = self.reverse_rnn(inputs, hidden)

        output = torch.cat((forward_output, reverse_output), dim=2)
        output = self.output_nn(output)
        hidden = torch.cat((forward_hidden, reverse_hidden), dim=2)
        hidden = self.hidden_nn(hidden)
        cell = torch.cat((forward_cell, reverse_cell), dim=2)
        cell = self.cell_nn(cell)

        return output, (hidden, cell)


class Encoder(nn.Module):
    def __init__(self, embedding_size, embedding_dim, rnn_dim, rnn_bias, pad_id, embedding_dropout=0, rnn_dropout=0,
                 dropout=0, n_layers=1, bidirectional=True, residual=True, weight_norm=True,
                 encoder_output_transformer=None, encoder_output_transformer_bias=None,
                 encoder_hidden_transformer=None, encoder_hidden_transformer_bias=None):
        super().__init__()
        self.embedding = nn.Embedding(embedding_size, embedding_dim, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.bidirectional = bidirectional

        # rnn cell
        cell = StackRNNCell(input_size=self.embedding.embedding_dim, hidden_size=rnn_dim, n_layers=n_layers,
                            bias=rnn_bias, dropout=rnn_dropout, residual=residual)
        if bidirectional:
            assert encoder_output_transformer and encoder_output_transformer_bias \
                   and encoder_hidden_transformer and encoder_hidden_transformer_bias, 'not input transformer parameter'
            self.rnn = BiRecurrent(cell, encoder_output_transformer, encoder_output_transformer_bias,
                                   encoder_hidden_transformer, encoder_hidden_transformer_bias)
        else:
            self.rnn = Recurrent(cell)

    def forward(self, enc_input):
        embedded = self.embedding_dropout(
            self.embedding(enc_input)
        )
        output, (hidden, cell) = self.rnn(embedded)
        output = self.dropout(output)
        return output, (hidden, cell)


class AttentionDecoder(nn.Module):
    def __init__(self, embedding_size, embedding_dim, rnn_dim, rnn_bias, pad_id, embedding_dropout=0, rnn_dropout=0,
                 dropout=0, n_layers=1, residual=True, weight_norm=True, attention_score_func='dot'):
        super().__init__()
        self.vocab_size = embedding_size
        self.embedding = nn.Embedding(embedding_size, embedding_dim, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = Attention(score_function=attention_score_func, hidden_size=rnn_dim)
        cell = StackAttentionCell(input_size=self.embedding.embedding_dim, hidden_size=rnn_dim,
                                  attention=self.attention, n_layers=n_layers, bias=rnn_bias,
                                  dropout=rnn_dropout, residual=residual)
        self.rnn = Recurrent(cell, with_attention=True)
        self.classifier = nn.Linear(rnn_dim, embedding_size)

    def forward(self, context, dec_input, hidden, get_attention=False):
        embedded = self.embedding(dec_input)
        if get_attention:
            output, attention, hidden = \
                self.rnn(inputs=embedded, hidden=hidden, context=context, get_attention=get_attention)
            output = self.dropout(output)
            output = self.classifier(output)
            return output, attention, hidden
        else:
            output, hidden = self.rnn(inputs=embedded, hidden=hidden, context=context, get_attention=get_attention)
            output = self.dropout(output)
            output = self.classifier(output)
            return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, seq_len, get_attention):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seq_len = seq_len
        self.get_attention = get_attention

    def forward(self, enc_input, dec_input, teacher_forcing_rate=0.5):
        context, hidden = self.encoder(enc_input)

        # teacher forcing ratio check
        if teacher_forcing_rate == 1.0:
            if self.get_attention:
                output, attention, _ = self.decoder(context=context, dec_input=dec_input, hidden=hidden,
                                                    get_attention=True)
                return output, attention
            else:
                output, _ = self.decoder(context=context, dec_input=dec_input, hidden=hidden,
                                         get_attention=False)
                return output
        else:
            outputs = []
            attentions = []

            dec_input_i = dec_input[:, 0].unsqueeze(dim=1)
            if self.get_attention:
                for i in range(1, self.seq_len+1):
                    output, attention, hidden = self.decoder(context=context, dec_input=dec_input_i, hidden=hidden,
                                                             get_attention=True)
                    _, indices = output.max(dim=2)

                    output = output.squeeze(dim=1)
                    attention = attention.squeeze(dim=2)
                    outputs.append(output)
                    attentions.append(attention)

                    if i != self.seq_len:
                        dec_input_i = \
                            dec_input[:, i].unsqueeze(dim=1) if random.random() < teacher_forcing_rate else indices

                outputs = torch.stack(outputs, dim=1)
                attentions = torch.stack(attentions, dim=2)
                return outputs, attentions
            else:
                for i in range(1, self.seq_len+1):
                    output, hidden = self.decoder(context=context, dec_input=dec_input_i, hidden=hidden,
                                                  get_attention=False)
                    _, indices = output.max(dim=2)

                    output = output.squeeze(dim=1)
                    outputs.append(output)

                    if i != self.seq_len:
                        dec_input_i = \
                            dec_input[:, i].unsqueeze(dim=1) if random.random() < teacher_forcing_rate else indices

                outputs = torch.stack(outputs, dim=1)
                return outputs


class Attention(nn.Module):
    def __init__(self, score_function, hidden_size):
        super().__init__()
        if score_function not in ['dot', 'general', 'concat']:
            raise NotImplemented('Not implemented {} attention score function '
                                 'you must selected [dot, general, concat]'.format(score_function))

        self.score_function = score_function
        if score_function == 'general':
            self.linear = nn.Linear(hidden_size, hidden_size)
        elif score_function == 'concat':
            self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
            self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, context, target):
        target = target.unsqueeze(dim=2)

        attention_score = None
        if self.score_function == 'dot':
            attention_score = torch.bmm(context, target)
        elif self.score_function == 'general':
            output = self.linear(context)
            attention_score = torch.bmm(output, target)
        elif self.score_function == 'concat':
            target = target.expand(-1, -1, context.size(1)).permute(0, 2, 1)
            pack = torch.cat((context, target), dim=2)
            output = self.linear1(pack)
            attention_score = self.linear2(output)

        attention_distribution = F.log_softmax(attention_score, dim=1)
        context = context * attention_distribution
        context_vector = context.sum(dim=1)

        return attention_distribution, context_vector
