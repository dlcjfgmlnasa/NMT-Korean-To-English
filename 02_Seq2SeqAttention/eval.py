# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from data_helper import split_sentence_with_ko, split_sentence_with_en, TranslationDataset, Voc


# matplotlib setting
plt.rcParams["font.family"] = 'NanumBarunGothic'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
translation_obj = TranslationDataset


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
