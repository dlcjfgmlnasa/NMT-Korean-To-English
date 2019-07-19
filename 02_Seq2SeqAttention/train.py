# -*- coding:utf-8 -*-
import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_helper import TranslationDataset, create_or_get_voc, create_or_get_word2vec, apply_word2vec_embedding_matrix
from model import EncoderRNN, Attention, DecoderAttentionRNN, Seq2SeqAttention
from tensorboardX import SummaryWriter


writer = SummaryWriter()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../Dataset', type=str)
    parser.add_argument('--word2vec_path', default='../Word2Vec', type=str)
    parser.add_argument('--rnn_sequence_size', default=20, type=int)
    parser.add_argument('--min_count', default=0, type=int)
    parser.add_argument('--max_count', default=8000, type=int)
    parser.add_argument('--embedding_size', default=200, type=int)
    parser.add_argument('--rnn_dim', default=128, type=int)
    parser.add_argument('--rnn_layer', default=1, type=int)
    parser.add_argument('--attention_method', default='general', choices=['dot', 'general', 'concat'], type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--model_path', default='./save_model/0_seq2seq.pth', type=str)
    args = parser.parse_args()
    return args


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculation_loss(out, tar, loss_func):
    out = out.view(-1, out.size(-1))        # => (batch_size * seq_len, voc_size)
    tar = tar.view(-1)                      # => (batch_size * seq_len)

    out = out.to(device)
    tar = tar.to(device)

    loss_ = loss_func(out, tar)
    return loss_


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


def train():
    args = get_args()
    x_train_path = os.path.join(args.data_path, 'train.ko')
    y_train_path = os.path.join(args.data_path, 'train.en')
    x_dev_path = os.path.join(args.data_path, 'dev.ko')
    y_dev_path = os.path.join(args.data_path, 'dev.en')

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

    # create or load word2vec model
    ko_word2vec, en_word2vec = create_or_get_word2vec(args.word2vec_path, x_train_path, y_train_path)

    # embedding matrix
    ko_embedding = nn.Embedding(ko_word_len, args.embedding_size)
    en_embedding = nn.Embedding(en_word_len, args.embedding_size)

    # initialize embedding matrix - apply word2vec embedding matrix
    ko_embedding = apply_word2vec_embedding_matrix(ko_word2vec, ko_embedding, ko_voc)
    en_embedding = apply_word2vec_embedding_matrix(en_word2vec, en_embedding, en_voc)

    # define model
    encoder = EncoderRNN(ko_embedding, args.rnn_sequence_size, args.rnn_dim, args.rnn_layer)
    attention = Attention(args.attention_method, args.rnn_dim)
    decoder = DecoderAttentionRNN(attention, en_embedding, args.rnn_dim, en_word_len, args.rnn_layer)
    model = Seq2SeqAttention(encoder, decoder)
    model.to(device)
    model.train()

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=en_voc.word2idx[en_voc.PAD])

    # if exist model => load model
    if os.path.exists(args.model_path):
        print(f'load model : {args.model_path}')
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # load train data & loader
    train_data = TranslationDataset(x_train_path, y_train_path, ko_voc, en_voc, args.rnn_sequence_size)
    # train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=1)

    # load dev data & loader
    dev_data = TranslationDataset(x_dev_path, y_dev_path, ko_voc, en_voc, args.rnn_sequence_size)
    # dev_loader = DataLoader(dev_data, shuffle=True, batch_size=int(dev_data.__len__() / 50))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=1)

    # Training
    start_time = time.time()
    global_step = 0
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
            output = model(train_enc_input, train_enc_length, train_dec_input)

            # get loss & accuracy
            loss = calculation_loss(output, train_dec_output, criterion)
            accuracy = calculation_accuracy(output, train_dec_input)

            writer.add_scalar('loss', loss.item(), global_step)
            writer.add_scalar('accuracy', accuracy.item(), global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Dev data loss calculation
            if i % 50 == 0 and i != 0:
                dev_avg_loss = 0
                dev_accuracy = 0
                count = 0

                for dev_enc_input, dev_enc_length, dev_dec_input, dev_dec_output in dev_loader:
                    dev_enc_length, sorted_idx = dev_enc_length.sort(0, descending=True)
                    dev_enc_input = dev_enc_input[sorted_idx]

                    dev_output = model(dev_enc_input, dev_enc_length, dev_dec_input)
                    dev_loss = calculation_loss(dev_output, dev_dec_output, criterion)
                    dev_accuracy = calculation_accuracy(dev_output, dev_dec_output)

                    dev_avg_loss += dev_loss.item()
                    dev_accuracy += dev_accuracy.item()
                    count += 1

                dev_avg_loss = dev_avg_loss / count
                dev_accuracy = dev_accuracy / count
                print('epoch : {0:2d} iter : {1:4d}  =>  train_loss : {2:4f}  dev_loss : {3:4f}  dev_accuracy: {4:4f}'.
                      format(epoch, i, loss.item(), dev_avg_loss, dev_accuracy))
            else:
                epoch_avg_train_loss += loss.item()
                print('epoch : {0:2d} iter : {1:4d}  =>  train_loss : {2:4f}  accuracy : {3:4f}'.
                      format(epoch, i, loss.item(), accuracy.item()))

            global_step += 1

        # save model
        if epoch % 5 == 0:
            base_path = os.path.dirname(args.model_path)
            full_path = os.path.join(base_path, str(epoch)+'_attention_seq2seq.pth')
            torch.save({
                'epoch': epoch+1,
                'ko_embedding': ko_embedding.state_dict(),
                'en_embedding': en_embedding.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_avg_train_loss / len(iter(train_loader))
            }, full_path)

    end_time = time.time()
    writer.close()

    print('Training ending... !! time : {}'.format(end_time - start_time))


if __name__ == '__main__':
    train()
