# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_helper import create_or_get_voc, RNNSeq2SeqDataset
from model import EncoderRNN, DecoderRNN, Seq2Seq


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../Dataset', type=str)
    parser.add_argument('--dictionary_path', default='../Dictionary', type=str)
    parser.add_argument('--rnn_sequence_size', default=30, type=int)
    parser.add_argument('--embedding_size', default=200, type=int)
    parser.add_argument('--rnn_dim', default=200, type=int)
    parser.add_argument('--rnn_layer', default=3, type=int)
    parser.add_argument('--model_path', default='./save_model/30_seq2seq.pth', type=str)
    return parser.parse_args()


def calculation_accuracy(out, tar):
    _, indices = out.max(dim=2)
    indices = indices.to(device)
    tar = tar.to(device)

    equal = indices.eq(tar)
    total = 1
    for i in equal.size():
        total *= i

    accuracy = torch.div(equal.sum().to(dtype=torch.float32), total)
    return accuracy


def load_voc(dictionary_path):
    # load vocabulary dictionary
    ko_voc, en_voc = create_or_get_voc(
        save_path=dictionary_path,
    )
    return ko_voc, en_voc


def load_model(model_path, ko_voc, en_voc, embedding_size, seq_len, rnn_dim, rnn_layer):
    # load seq2seq model
    checkpoint = torch.load(model_path)
    ko_word_len = len(ko_voc.word2idx)
    en_word_len = len(en_voc.word2idx)

    # embedding matrix
    ko_embedding = nn.Embedding(ko_word_len, embedding_size)
    en_embedding = nn.Embedding(en_word_len, embedding_size)

    # define model
    encoder = EncoderRNN(ko_embedding, seq_len, rnn_dim, rnn_layer)
    decoder = DecoderRNN(en_embedding, rnn_dim, en_word_len, rnn_layer)
    model = Seq2Seq(encoder, decoder)
    model.to(device)

    # load model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def translation():
    args = get_args()
    test_x_path = os.path.join(args.data_path, 'test.ko')
    test_y_path = os.path.join(args.data_path, 'test.en')

    ko_voc, en_voc = load_voc(args.dictionary_path)
    model = load_model(args.model_path, ko_voc, en_voc, args.embedding_size, args.rnn_sequence_size,
                       args.rnn_dim, args.rnn_layer)

    # load test data & loader
    test_data = RNNSeq2SeqDataset(test_x_path, test_y_path, ko_voc, en_voc, args.rnn_sequence_size)
    test_loader = DataLoader(test_data, batch_size=1)

    for enc_input, enc_length, dec_input, dec_output in test_loader:
        enc_length, sorted_idx = enc_length.sort(0, descending=True)
        enc_input = enc_input[sorted_idx]
        output = model(enc_input, enc_length, dec_input)
        accuracy = calculation_accuracy(output, dec_output)

        output = output.squeeze(0)
        value, indices = output.max(1)
        target_value = [ko_voc.idx2word[int(idx)] for idx in enc_input[0]]
        translation_values = [en_voc.idx2word[int(idx)] for idx in indices]
        print(' '.join([value for value in target_value if value not in ['__END__', '__UNK__', '__PAD__']]))
        print(' '.join([value for value in translation_values if value not in ['__END__', '__UNK__', '__PAD__']]))


translation()

