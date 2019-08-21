# -*- coding:utf-8 -*-
import os
import sys; sys.path.append('..')
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from data_helper import create_or_get_voc_v2, RNNSeq2SeqDatasetV2
from model import Encoder, AttentionDecoder, Seq2Seq
from tensorboardX import SummaryWriter
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/NanumBarunGothic.ttf").get_name()
rc('font', family=font_name)
plt.rcParams.update({'font.size': 7})


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.ko_voc, self.en_voc = self.get_voc()
        self.train_loader = self.get_train_loader()
        self.val_loader = self.get_val_loader()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.en_voc['<pad>'], reduction='mean'
        )
        self.writer = SummaryWriter()
        self.train()

    def train(self):
        encoder_parameter = self.encoder_parameter()
        decoder_parameter = self.decoder_parameter()

        encoder = Encoder(**encoder_parameter)
        decoder = AttentionDecoder(**decoder_parameter)
        model = Seq2Seq(encoder, decoder, self.args.sequence_size, self.args.get_attention)
        model.train()
        model.to(device)

        optimizer = opt.Adam(model.parameters(), lr=self.args.learning_rate)

        epoch_step = len(self.train_loader) + 1
        total_step = self.args.epochs * epoch_step
        teacher_forcing_ratios = self.cal_teacher_forcing_ratio(total_step)

        step = 0
        attention = None

        for epoch in range(self.args.epochs):
            for i, data in enumerate(self.train_loader, 0):
                try:
                    src_input, trg_input, trg_output = data

                    if self.args.get_attention:
                        output, attention = model(src_input, trg_input, teacher_forcing_rate=teacher_forcing_ratios[i])
                    else:
                        output = model(src_input, trg_input, teacher_forcing_rate=teacher_forcing_ratios[i])

                    # Get loss & accuracy
                    loss, accuracy = self.loss_accuracy(output, trg_output)

                    # Training Log
                    if step % self.args.train_step_print == 0:
                        self.writer.add_scalar('train/loss', loss.item(), step)
                        self.writer.add_scalar('train/accuracy', accuracy.item(), step)

                        print('[Train] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                              '=>  loss : {5:10f}  accuracy : {6:12f}'
                              .format(epoch, i, epoch_step, step, total_step, loss.item(), accuracy.item()))

                    # Validation Log
                    if step % self.args.val_step_print == 0:
                        with torch.no_grad():
                            val_loss, val_accuracy = self.val(model, teacher_forcing_ratio=teacher_forcing_ratios[i])
                            self.writer.add_scalar('val/loss', val_loss, step)
                            self.writer.add_scalar('val/accuracy', val_accuracy, step)

                            print('[ Val ] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                                  '=>  loss : {5:10f}  accuracy : {6:12f}'
                                  .format(epoch, i, epoch_step, step, total_step, val_loss, val_accuracy))

                    # Save Model Point
                    if step % self.args.step_save == 0:
                        if self.args.get_attention:
                            self.plot_attention(step, src_input, trg_input, attention)
                        self.model_save(model=model, optimizer=optimizer, epoch=epoch, step=step)

                    # Optimizer
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # If KeyBoard Interrupt Save Model
                except KeyboardInterrupt:
                    self.model_save(model=model, optimizer=optimizer, epoch=epoch, step=step)

    def val(self, model, teacher_forcing_ratio):
        total_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            count = 0
            for data in self.val_loader:
                src_input, trg_input, trg_output = data
                output = model(src_input, trg_input, teacher_forcing_rate=teacher_forcing_ratio)
                if isinstance(output, tuple):
                    output = output[0]
                loss, accuracy = self.loss_accuracy(output, trg_output)
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                count += 1
        avg_loss = total_loss / count
        avg_accuracy = total_accuracy / count
        return avg_loss, avg_accuracy

    def loss_accuracy(self, output, target):
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        output = F.log_softmax(output, -1)

        loss = self.criterion(output, target)
        _, indices = output.max(-1)
        invalid_targets = target.eq(self.en_voc['<pad>'])
        accuracy = indices.eq(target).masked_fill_(
            invalid_targets, 0).long().sum()

        return loss, accuracy

    def model_save(self, model, optimizer, epoch, step):
        model_name = '{0:06d}_attention_seq2seq.pth'.format(step)
        model_path = os.path.join(self.args.model_path, model_name)
        torch.save({
            'epoch': epoch,
            'steps': step,
            'seq_len': self.args.sequence_size,
            'encoder_parameter': self.encoder_parameter(),
            'decoder_parameter': self.decoder_parameter(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_path)

    def cal_teacher_forcing_ratio(self, total_step):
        if self.args.learning_method == 'Teacher_Forcing':
            teacher_forcing_ratios = [1.0 for _ in range(total_step)]
        elif self.args.learning_method == 'Scheduled_Sampling':
            import numpy as np
            teacher_forcing_ratios = np.linspace(0.0, 1.0, num=total_step)[::-1]
        else:
            raise NotImplemented('learning method must choice [Teacher_Forcing, Scheduled_Sampling]')
        return teacher_forcing_ratios

    def get_voc(self):
        try:
            ko_voc, en_voc = create_or_get_voc_v2(save_path=self.args.dictionary_path)
        except OSError:
            src_train_path = os.path.join(self.args.data_path, self.args.src_train_filename)
            trg_train_path = os.path.join(self.args.data_path, self.args.trg_train_filename)
            ko_voc, en_voc = create_or_get_voc_v2(save_path=self.args.dictionary_path,
                                                  ko_corpus_path=src_train_path,
                                                  en_corpus_path=trg_train_path)
        return ko_voc, en_voc

    def get_train_loader(self):
        x_train_path = os.path.join(self.args.data_path, self.args.src_train_filename)
        y_train_path = os.path.join(self.args.data_path, self.args.trg_train_filename)
        train_dataset = RNNSeq2SeqDatasetV2(x_train_path,
                                            y_train_path,
                                            self.ko_voc,
                                            self.en_voc,
                                            self.args.sequence_size)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        return train_loader

    def get_val_loader(self):
        x_val_path = os.path.join(self.args.data_path, self.args.src_val_filename)
        y_val_path = os.path.join(self.args.data_path, self.args.trg_val_filename)
        val_dataset = RNNSeq2SeqDatasetV2(x_val_path,
                                          y_val_path,
                                          self.ko_voc,
                                          self.en_voc,
                                          self.args.sequence_size)
        val_loader = DataLoader(val_dataset, batch_size=int(len(val_dataset) / 10), shuffle=False)
        return val_loader

    def plot_attention(self, step, src_input, trg_input, attention):
        filename = '{0:06d}_step'.format(step)
        filepath = os.path.join(self.args.img_path, filename)
        os.mkdir(filepath)

        def replace_pad(words):
            return [word if word != '<pad>' else '' for word in words]

        with torch.no_grad():
            src_input = src_input.to('cpu')
            trg_input = trg_input.to('cpu')
            attention = attention.to('cpu')

            sample = [i for i in range(src_input.shape[0]-1)]
            sample = random.sample(sample, self.args.plot_count)

            for num, i in enumerate(sample):
                src, trg = src_input[i], trg_input[i]
                src_word = replace_pad([self.ko_voc.IdToPiece(word.item()) for word in src])
                trg_word = replace_pad([self.en_voc.IdToPiece(word.item()) for word in trg])

                fig = plt.figure()
                ax = fig.add_subplot(111)
                cax = ax.matshow(attention[i].data, cmap='bone')
                fig.colorbar(cax)

                ax.set_xticklabels(trg_word, rotation=90)
                ax.set_yticklabels(src_word)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                fig.savefig(fname=os.path.join(filepath, 'attention-{}.jpg'.format(num)))

    def encoder_parameter(self):
        param = {
            'embedding_size': 4000,
            'embedding_dim': self.args.embedding_dim,
            'pad_id': self.ko_voc['<pad>'],
            'embedding_dropout': self.args.encoder_embedding_dropout,
            'rnn_dropout': self.args.encoder_rnn_dropout,
            'dropout': self.args.encoder_dropout,
            'rnn_dim': self.args.encoder_rnn_dim,
            'rnn_bias': True,
            'n_layers': self.args.encoder_n_layers,
            'bidirectional': self.args.encoder_bidirectional_used,
            'residual': self.args.encoder_residual_used,
            'weight_norm': self.args.encoder_weight_norm_used,
            'encoder_output_transformer': self.args.encoder_output_transformer,
            'encoder_output_transformer_bias': self.args.encoder_output_transformer_bias,
            'encoder_hidden_transformer': self.args.encoder_hidden_transformer,
            'encoder_hidden_transformer_bias': self.args.encoder_hidden_transformer_bias,
        }
        return param

    def decoder_parameter(self):
        param = {
            'embedding_size': 4000,
            'embedding_dim': self.args.embedding_dim,
            'pad_id': self.en_voc['<pad>'],
            'embedding_dropout': self.args.decoder_embedding_dropout,
            'rnn_dropout': self.args.decoder_rnn_dropout,
            'dropout': self.args.decoder_dropout,
            'rnn_dim': self.args.decoder_rnn_dim,
            'rnn_bias': True,
            'n_layers': self.args.decoder_n_layers,
            'residual': self.args.decoder_residual_used,
            'attention_score_func': self.args.attention_score
        }
        return param


class Translation(object):
    def __init__(self,
                 checkpoint,
                 dictionary_path,
                 get_attention
                 ):
        self.checkpoint = torch.load(checkpoint)
        self.seq_len = self.checkpoint['seq_len']
        self.batch_size = 100
        self.get_attention = get_attention

        self.ko_voc, self.en_voc = create_or_get_voc_v2(save_path=dictionary_path)
        self.model = self.model_load()

    def transform(self, sentence: str) -> (str, torch.Tensor):
        src_input = self.src_input(sentence)
        trg_input = self.trg_input()

        attention = None
        if self.get_attention:
            output, attention = self.model(src_input, trg_input, teacher_forcing_rate=0.0)
        else:
            output = self.model(src_input, trg_input, teacher_forcing_rate=0.0)
        _, indices = output.max(dim=2)
        result = self.tensor2sentence(indices)[0]

        return result, attention

    def batch_transform(self, sentence_list: list) -> (list, torch.Tensor):
        if len(sentence_list) > self.batch_size:
            raise ValueError('You must sentence size less than {}'.format(self.batch_size))

        src_inputs = torch.stack([self.src_input(sentence) for sentence in sentence_list]).squeeze(dim=1)
        trg_inputs = torch.stack([self.trg_input() for _ in sentence_list]).squeeze(dim=1)

        attention = None
        if self.get_attention:
            output, attention = self.model(src_inputs, trg_inputs, teacher_forcing_rate=0.0)
        else:
            output = self.model(src_inputs, trg_inputs, teacher_forcing_rate=0.0)

        _, indices = output.max(dim=2)
        result = self.tensor2sentence(indices)
        return result, attention

    def tensor2sentence(self, indices: torch.Tensor) -> list:
        result = []

        for idx_list in indices:
            translation_sentence = []
            for idx in idx_list:
                word = self.en_voc.IdToPiece(idx.item())
                if word == '</s>':
                    break
                translation_sentence.append(word)
            translation_sentence = ''.join(translation_sentence).replace('▁', ' ').strip()
            result.append(translation_sentence)

        return result

    def src_input(self, sentence):
        idx_list = self.ko_voc.EncodeAsIds(sentence)
        idx_list = self.padding(idx_list, self.ko_voc['<pad>'])
        return torch.tensor([idx_list]).to(device)

    def trg_input(self):
        idx_list = [self.en_voc['<s>']]
        return torch.tensor([idx_list]).to(device)

    def padding(self, idx_list, padding_id):
        length = len(idx_list)
        if length < self.seq_len:
            idx_list = idx_list + [padding_id for _ in range(self.seq_len - len(idx_list))]
        else:
            idx_list = idx_list[:self.seq_len]
        return idx_list

    def model_load(self):
        encoder = Encoder(**self.checkpoint['encoder_parameter'])
        decoder = AttentionDecoder(**self.checkpoint['decoder_parameter'])
        model = Seq2Seq(encoder, decoder, self.seq_len, self.get_attention)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
