# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from data_helper import split_sentence_with_ko, RNNSeq2SeqDataset, Voc, create_or_get_voc
from model import EncoderRNN, Attention, DecoderAttentionRNN, Seq2SeqAttention


# matplotlib setting
plt.rcParams["font.family"] = 'NanumBarunGothic'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
translation_obj = RNNSeq2SeqDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../Dataset', type=str)
    parser.add_argument('--dictionary_path', default='../Dictionary', type=str)
    parser.add_argument('--word2vec_path', default='../Word2Vec', type=str)
    parser.add_argument('--rnn_sequence_size', default=30, type=int)
    parser.add_argument('--min_count', default=3, type=int)
    parser.add_argument('--max_count', default=100000, type=int)
    parser.add_argument('--embedding_size', default=200, type=int)
    parser.add_argument('--rnn_dim', default=200, type=int)
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--dec_layers', default=3, type=int)
    parser.add_argument('--rnn_layers', default=5, type=int)
    parser.add_argument('--rnn_dropout_rate', default=0.5, type=float)
    parser.add_argument('--use_residual', default=True, type=bool)
    parser.add_argument('--attention_method', default='general', choices=['dot', 'general', 'concat'], type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--model_path', default='./save_model/0_attention_seq2seq.pth', type=str)
    args = parser.parse_args()
    return args


def load_voc(dictionary_path):
    # load vocabulary dictionary
    ko_voc, en_voc = create_or_get_voc(
        save_path=dictionary_path,
    )
    return ko_voc, en_voc


def load_model(model_path, ko_voc, en_voc, embedding_size):
    args = get_args()
    # load seq2seq model
    checkpoint = torch.load(model_path)
    ko_word_len = len(ko_voc.word2idx)
    en_word_len = len(en_voc.word2idx)

    # embedding matrix
    ko_embedding = nn.Embedding(ko_word_len, embedding_size)
    en_embedding = nn.Embedding(en_word_len, embedding_size)

    # define model
    encoder = EncoderRNN(
        ko_embedding, args.rnn_sequence_size, args.rnn_dim, args.rnn_layers, args.rnn_dropout_rate, args.use_residual)
    attention = Attention(args.attention_method, args.rnn_dim)
    decoder = DecoderAttentionRNN(attention, en_embedding, args.rnn_dim, en_word_len, args.rnn_layers,
                                  args.rnn_dropout_rate, args.use_residual)
    model = Seq2SeqAttention(encoder, decoder)
    model.to(device)

    # load model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def translation(model: nn.Module, src_input: str, ko_voc: Voc, en_voc: Voc,
                sequence_size: int, show_attention=True):
    # prepare => sentence to tensor
    src_words = split_sentence_with_ko(src_input)
    src_words, src_length = padding(src_words, ko_voc, sequence_size)
    src_inputs = translation_obj.word2idx(src_words, ko_voc)

    trg_words = [en_voc.STR]
    trg_words, _ = padding(trg_words, en_voc, sequence_size)
    trg_inputs = translation_obj.word2idx(trg_words, en_voc)

    src_inputs, src_length, trg_inputs = \
        torch.tensor(src_inputs).to(device), torch.tensor(src_length).to(device), torch.tensor(trg_inputs).to(device)

    # un squeeze
    src_inputs = src_inputs.unsqueeze(dim=0)
    src_length = src_length.unsqueeze(dim=0)
    trg_inputs = trg_inputs.unsqueeze(dim=0)

    output, attention_weight = model(src_inputs, src_length, trg_inputs)
    _, indices = output.max(dim=2)

    translation_words = [en_voc.idx2word[int(idx)] for idx in indices[0]]
    translation_sentence = ' '.join([word for word in translation_words
                                     if word not in [ko_voc.STR, ko_voc.END, ko_voc.UNK, ko_voc.PAD]])
    if show_attention:
        fig = plot_attention(src_words, translation_words, attention_weight)
        return translation_sentence, fig
    return translation_sentence


def plot_attention(src_words, trg_words, attention_weights):
    attention_weights = attention_weights.to('cpu')
    attention_weights = attention_weights.squeeze(dim=0)
    attention_weights = attention_weights.data

    # graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_weights.data, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(src_words, rotation=90)
    ax.set_yticklabels(trg_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    return fig


def padding(words, voc, sequence_size):
    if len(words) < sequence_size:
        length = len(words)
        words += [voc.PAD] * (sequence_size - len(words))
    else:
        words = words[:sequence_size]
        length = len(words)
    return words, length


if __name__ == '__main__':
    args = get_args()
    ko_voc, en_voc = load_voc(args.dictionary_path)
    model = load_model(args.model_path, ko_voc, en_voc, args.embedding_size)
    input_ = '그리고 미래 공항과 우주공항에 대한 개발된 근본적인 새로운 생각들이 있다.'

    result = translation(model, input_, ko_voc, en_voc, args.rnn_sequence_size, False)
    print(result)

