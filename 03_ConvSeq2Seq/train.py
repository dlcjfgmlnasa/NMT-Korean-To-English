# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division


import os
import argparse
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from data_helper import ConvSeq2SeqDataset, create_or_get_voc
from model import Encoder, Decoder, Seq2Seq

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../Dataset', type=str)
    parser.add_argument('--sequence_size', default=30, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--embedding_dims', default=500, type=int)
    parser.add_argument('--encoder_n_layers', default=10, type=int)
    parser.add_argument('--decoder_n_layers', default=5, type=int)
    parser.add_argument('--encoder_dropout_rate', default=0.5, type=int)
    parser.add_argument('--decoder_dropout_rate', default=0.5, type=int)
    parser.add_argument('--hidden_size', default=200, type=int)
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--min_count', default=3, type=int)
    parser.add_argument('--max_count', default=100000, type=int)
    return parser.parse_args()


def train():
    args = get_args()
    x_train_path = os.path.join(args.data_path, 'train.ko')
    y_train_path = os.path.join(args.data_path, 'train.en')

    # create or load vocabulary
    ko_voc, en_voc = create_or_get_voc(
        x_train_path,
        y_train_path,
        save_path='../Dictionary',
        min_count=args.min_count,
        max_count=args.max_count
    )
    ko_word_len = len(ko_voc.word2idx)
    en_word_len = len(en_voc.word2idx)

    train_data = ConvSeq2SeqDataset(x_train_path, y_train_path, ko_voc, en_voc, args.sequence_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    dev_data = ConvSeq2SeqDataset(x_train_path, y_train_path, ko_voc, en_voc, args.sequence_size)
    dev_loader = DataLoader(dev_data, batch_size=1)

    encoder = Encoder(ko_word_len, args.embedding_dims, args.encoder_n_layers, args.hidden_size, args.kernel_size)
    decoder = Decoder(en_word_len, args.embedding_dims, args.decoder_n_layers, args.hidden_size, args.kernel_size,
                      en_voc.word2idx[en_voc.PAD])
    model = Seq2Seq(encoder, decoder)
    model.to(device)
    model.train()

    optimizer = opt.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=en_voc.word2idx[en_voc.PAD])

    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            train_enc_input, train_dec_input, train_dec_output = data
            output, attention_score = model(train_enc_input, train_dec_input, train_dec_output)
            del attention_score
            output = output.contiguous().view(-1, output.shape[-1])
            target = train_dec_output.contiguous().view(-1)
            loss = criterion(output, target)

            # optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.item())


if __name__ == '__main__':
    train()
