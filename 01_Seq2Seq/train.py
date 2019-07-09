# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division


import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import EncoderRNN, DecoderRNN, Seq2Seq
from data_helper import create_or_get_voc, TranslationDataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../Dataset', type=str)
    parser.add_argument('--rnn_sequence_size', default=20, type=int)
    parser.add_argument('--min_count', default=0, type=int)
    parser.add_argument('--max_count', default=8000, type=int)
    parser.add_argument('--embedding_size', default=200, type=int)
    parser.add_argument('--rnn_dim', default=50, type=int)
    parser.add_argument('--rnn_layer', default=1, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model_path', default='./save_model/0_seq2seq.pth', type=str)
    args = parser.parse_args()
    return args


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculation_loss(out, tar, loss_func):
    out = out[:, :-1].contiguous()          # => (batch_size, seq_len-1, voc_size)
    tar = tar[:, 1:].contiguous()           # => (batch_size, seq_len-1)
    out = out.view(-1, out.size(-1))        # => (batch_size * seq_len-1, voc_size)
    tar = tar.view(-1)                      # => (batch_size * seq_len-1)

    out = out.to(device)
    tar = tar.to(device)

    loss_ = loss_func(out, tar)
    return loss_


def train():
    args = get_args()

    x_train_path = os.path.join(args.data_path, 'train.ko')
    y_train_path = os.path.join(args.data_path, 'train.en')
    x_dev_path = os.path.join(args.data_path, 'dev.ko')
    y_dev_path = os.path.join(args.data_path, 'dev.en')

    ko_voc, en_voc = create_or_get_voc(
        x_train_path,
        y_train_path,
        save_path='../Dictionary',
        min_count=args.min_count,
        max_count=args.max_count
    )
    ko_word_len = len(ko_voc.word2idx)
    en_word_len = len(en_voc.word2idx)

    # embedding matrix
    ko_embedding = nn.Embedding(ko_word_len, args.embedding_size)
    en_embedding = nn.Embedding(en_word_len, args.embedding_size)

    # define model
    encoder = EncoderRNN(ko_embedding, args.rnn_dim, args.rnn_layer)
    decoder = DecoderRNN(en_embedding, args.rnn_dim, en_word_len, args.rnn_layer)
    model = Seq2Seq(encoder, decoder)
    model = model.to(device)

    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=en_voc.word2idx[en_voc.PAD])

    # if exist model => load model
    if os.path.exists(args.model_path):
        print(f'load model : {args.model_path}')
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Train data & loader
    train_data = TranslationDataset(x_train_path, y_train_path, ko_voc, en_voc, args.rnn_sequence_size)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=0)

    # Dev data & loader
    dev_data = TranslationDataset(x_dev_path, y_dev_path, ko_voc, en_voc, args.rnn_sequence_size)
    dev_loader = DataLoader(dev_data, batch_size=int(dev_data.__len__() / 50))

    # Training
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_avg_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            # train_enc_input  => (batch_size, sequence_size)
            # train_enc_length => (batch_size)
            # train_dec_input  => (batch_size, sequence_size)
            # train_dec_output => (batch_size, sequence_size)
            train_enc_input, train_enc_length, train_dec_input, train_dec_output = data
            train_enc_length, sorted_idx = train_enc_length.sort(0, descending=True)
            train_enc_input = train_enc_input[sorted_idx]

            # nn forward
            output = model(train_enc_input, train_enc_length, train_dec_input)  # => (batch_size, seq_len, voc_size)

            # get loss
            loss = calculation_loss(output, train_dec_output, criterion)

            # optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Dev data loss calculation
            if i % 50 == 0 and i != 0:
                dev_avg_loss = 0
                count = 0
                for dev_enc_input, dev_enc_length, dev_dec_input, dev_dec_output in dev_loader:
                    dev_enc_length, sorted_idx = dev_enc_length.sort(0, descending=True)
                    dev_enc_input = dev_enc_input[sorted_idx]

                    dev_output = model(dev_enc_input, dev_enc_length, dev_dec_input)
                    dev_loss = calculation_loss(dev_output, dev_dec_output, criterion)
                    dev_avg_loss += dev_loss.item()
                    count += 1

                dev_avg_loss = dev_avg_loss / count
                print('epoch : {0:2d} iter : {1:4d}  =>  train_loss : {2:4f}  dev_loss : {3:4f}'.format(
                    epoch, i, loss.item(), dev_avg_loss))
            else:
                epoch_avg_train_loss += loss.item()
                print('epoch : {0:2d} iter : {1:4d}  =>  train_loss : {2:4f}'.format(epoch, i, loss.item()))

        # save model
        if epoch % 5 == 0:
            base_path = os.path.dirname(args.model_path)
            full_path = os.path.join(base_path, str(epoch)+'_seq2seq.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_avg_train_loss / len(iter(train_loader))
            }, full_path)

    end_time = time.time()
    print(end_time - start_time)


if __name__ == '__main__':
    train()
