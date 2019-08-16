# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import os
import re
import time
import shutil
import torch
import pickle
from abc import *
from konlpy.tag import Okt
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import sentencepiece as spm

okt = Okt()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Voc(object):
    def __init__(self):
        self.STR = '__STR__'      # start
        self.END = '__END__'      # end
        self.PAD = '__PAD__'      # pad
        self.UNK = '__UNK__'      # unknown

        self.words = set()
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}
        self._add_spc_key()

    def _add_spc_key(self):
        spc_keys = [self.PAD, self.STR, self.END, self.UNK]
        self.words.update(spc_keys)
        for count, word in enumerate(spc_keys):
            self.word2idx[word] = count
            self.idx2word[count] = word

    @staticmethod
    def split_func(lng):
        if lng == 'ko':
            return split_sentence_with_ko
        elif lng == 'en':
            return split_sentence_with_en

    def add_sentences(self, sentences, lng):
        if lng not in ['ko', 'en']:
            raise NotImplementedError('voc supported only [ko, en]')

        for sentence in sentences:
            self.words.update(self.split_func(lng)(sentence))
            # make word2count
            for word in self.split_func(lng=lng)(sentence):
                if word in self.word2count:
                    self.word2count[word] += 1
                else:
                    self.word2count[word] = 1

        # make word2idx, idx2word
        words = sorted(list(self.words.difference({self.PAD, self.STR, self.END, self.UNK})))
        for i, word in enumerate(words):
            i = i + 4
            self.word2idx[word] = i
            self.idx2word[i] = word

    def trim(self, min_count, max_count):
        trim_words = set()

        for word, count in self.word2count.items():
            if not min_count <= count:
                trim_words.add(word)
        self.words = self.words - trim_words
        self.words = self.words.difference({self.PAD, self.STR, self.END, self.UNK})

        # make word2idx, idx2word
        words = sorted(list(self.words))
        words = [self.PAD, self.STR, self.END, self.UNK] + words
        self.word2idx = {}
        self.idx2word = {}
        for i, word in enumerate(words):
            self.word2idx[word] = i
            self.idx2word[i] = word


def clean_str(string):
    string = string.strip()
    string = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', string)
    return string.strip().lower()


def split_sentence_with_ko(sentence):
    sentence = clean_str(sentence)
    words = [line[0] for line in okt.pos(sentence, norm=True)]
    return words


def split_sentence_with_en(sentence):
    sentence = clean_str(sentence)
    words = sentence.split(' ')
    return words


def create_or_get_voc(ko_data_path=None, en_data_path=None, min_count=0, max_count=8000, save_path=None):
    ko_voc_path = os.path.join(save_path, 'ko_voc.pkl')
    en_voc_path = os.path.join(save_path, 'en_voc.pkl')

    if os.path.exists(ko_voc_path) and os.path.join(en_voc_path):
        start_time = time.time()
        with open(ko_voc_path, 'rb') as f:
            ko_voc = pickle.load(f)

        with open(en_voc_path, 'rb') as f:
            en_voc = pickle.load(f)

        end_time = time.time()
        print('load dictionary time : {:.3f}sec'.format(end_time - start_time))
        return ko_voc, en_voc
    else:
        ko_lines = open(ko_data_path, 'r', encoding='utf-8').readlines()
        en_lines = open(en_data_path, 'r', encoding='utf-8').readlines()

        start_time = time.time()

        ko_voc = Voc()
        ko_voc.add_sentences(ko_lines, lng='ko')
        ko_voc.trim(min_count, max_count)

        en_voc = Voc()
        en_voc.add_sentences(en_lines, lng='en')
        en_voc.trim(min_count, max_count)

        with open(ko_voc_path, 'wb') as f:
            pickle.dump(ko_voc, f, pickle.HIGHEST_PROTOCOL)

        with open(en_voc_path, 'wb') as f:
            pickle.dump(en_voc, f, pickle.HIGHEST_PROTOCOL)

        end_time = time.time()
        print('make dictionary time : {:.3f}sec'.format(end_time - start_time))
        return ko_voc, en_voc


def create_or_get_voc_v2(save_path, ko_corpus_path=None, en_corpus_path=None, ko_vocab_size=8000, en_vocab_size=8000):
    ko_corpus_prefix = 'ko_corpus'
    en_corpus_prefix = 'en_corpus'

    if ko_corpus_path and en_corpus_path:
        templates = '--input={} --model_prefix={} --vocab_size={} ' \
                    '--bos_id=0 --eos_id=1 --unk_id=2 --pad_id=3'
        ko_model_train_cmd = templates.format(ko_corpus_path, ko_corpus_prefix, ko_vocab_size)
        en_model_train_cmd = templates.format(en_corpus_path, en_corpus_prefix, en_vocab_size)

        spm.SentencePieceTrainer.Train(ko_model_train_cmd)
        spm.SentencePieceTrainer.Train(en_model_train_cmd)

        shutil.move(ko_corpus_prefix + '.model', save_path)
        shutil.move(ko_corpus_prefix + '.vocab', save_path)
        shutil.move(en_corpus_prefix + '.model', save_path)
        shutil.move(en_corpus_prefix + '.vocab', save_path)
    ko_sp = spm.SentencePieceProcessor()
    en_sp = spm.SentencePieceProcessor()
    ko_sp.load(os.path.join(save_path, ko_corpus_prefix+'.model'))
    en_sp.load(os.path.join(save_path, en_corpus_prefix+'.model'))
    return ko_sp, en_sp


def create_or_get_word2vec(save_path, ko_data_path=None, en_data_path=None,
                           embedding_size=200, window_size=5,
                           min_count=0, max_count=8000):
    ko_word2vec_path = os.path.join(save_path, 'ko_word2vec.model')
    en_word2vec_path = os.path.join(save_path, 'en_word2vec.model')

    if not os.path.exists(ko_word2vec_path) and not os.path.exists(en_word2vec_path):
        ko_lines = open(ko_data_path, 'r', encoding='utf-8').readlines()
        en_lines = open(en_data_path, 'r', encoding='utf-8').readlines()
        ko_lines = [split_sentence_with_ko(line) for line in ko_lines]
        en_lines = [split_sentence_with_en(line) for line in en_lines]

        ko_model = Word2Vec(ko_lines, size=embedding_size, window=window_size, workers=4,
                            min_count=min_count, iter=100)
        en_model = Word2Vec(en_lines, size=embedding_size, window=window_size, workers=4,
                            min_count=min_count, iter=100)
        ko_model.save(ko_word2vec_path)
        en_model.save(en_word2vec_path)
        return ko_model, en_model
    else:
        ko_model = KeyedVectors.load(ko_word2vec_path)
        en_model = KeyedVectors.load(en_word2vec_path)
        return ko_model, en_model


def apply_word2vec_embedding_matrix(word2vec_model, embedding_matrix, voc):
    num_embeddings = embedding_matrix.num_embeddings
    embedding_dim = embedding_matrix.embedding_dim

    word2vec_embedding_matrix = torch.randn(num_embeddings, embedding_dim)
    for idx, word in voc.idx2word.items():
        try:
            word_matrix = torch.from_numpy(word2vec_model.wv[word])
            word2vec_embedding_matrix[idx] = word_matrix
        except KeyError:
            pass
    embedding_matrix = embedding_matrix.from_pretrained(word2vec_embedding_matrix)
    return embedding_matrix


class TranslationDataset(Dataset, metaclass=ABCMeta):
    # Translation Dataset abstract class
    def __init__(self, x_path, y_path, ko_voc, en_voc, sequence_size):
        self.x = open(x_path, 'r', encoding='utf-8').readlines()
        self.y = open(y_path, 'r', encoding='utf-8').readlines()
        self.ko_voc = ko_voc
        self.en_voc = en_voc
        self.sequence_size = sequence_size

    def __len__(self):
        if len(self.x) != len(self.y):
            raise IndexError('not equal x_path, y_path line size')
        return len(self.x)

    @abstractmethod
    def encoder_input_to_vector(self, sentence: str): pass
    @abstractmethod
    def decoder_input_to_vector(self, sentence: str): pass
    @abstractmethod
    def decoder_output_to_vector(self, sentence: str): pass


class RNNSeq2SeqDataset(TranslationDataset):
    # for Seq2Seq model & Seq2Seq attention model
    def __init__(self, x_path, y_path, ko_voc, en_voc, sequence_size):
        super().__init__(x_path, y_path, ko_voc, en_voc, sequence_size)

    def __getitem__(self, idx):
        encoder_input, encoder_length = self.encoder_input_to_vector(self.x[idx])
        decoder_input = self.decoder_input_to_vector(self.y[idx])
        decoder_output = self.decoder_output_to_vector(self.y[idx])

        return encoder_input, encoder_length, decoder_input, decoder_output

    def encoder_input_to_vector(self, sentence):
        words = split_sentence_with_ko(sentence)
        words, length = self.padding(words, self.ko_voc)
        idx_list = self.word2idx(words, self.ko_voc)

        return torch.tensor(idx_list).to(device), torch.tensor(length).to(device)

    def decoder_input_to_vector(self, sentence):
        words = split_sentence_with_en(sentence)
        words.insert(0, self.en_voc.STR)
        words, _ = self.padding(words, self.en_voc)
        idx_list = self.word2idx(words, self.en_voc)

        return torch.tensor(idx_list).to(device)

    def decoder_output_to_vector(self, sentence):
        words = split_sentence_with_en(sentence)
        words.append(self.en_voc.END)
        words, _ = self.padding(words, self.en_voc)
        idx_list = self.word2idx(words, self.en_voc)

        return torch.tensor(idx_list).to(device)

    def padding(self, words, voc):
        if len(words) < self.sequence_size:
            length = len(words)
            words += [voc.PAD] * (self.sequence_size - len(words))
        else:
            words = words[:self.sequence_size]
            length = len(words)
        return words, length

    @staticmethod
    def word2idx(words, voc):
        idx_list = []
        for word in words:
            try:
                idx_list.append(voc.word2idx[word])
            except KeyError:
                idx_list.append(voc.word2idx[voc.UNK])
        return idx_list


class ConvSeq2SeqDataset(TranslationDataset):
    # for Convolution Seq2Seq model
    def __init__(self, x_path, y_path, ko_voc, en_voc, sequence_size):
        super().__init__(x_path, y_path, ko_voc, en_voc, sequence_size)

    def __getitem__(self, idx):
        encoder_input = self.encoder_input_to_vector(self.x[idx])
        decoder_input = self.decoder_input_to_vector(self.y[idx])
        decoder_output = self.decoder_output_to_vector(self.y[idx])
        return encoder_input, decoder_input, decoder_output

    def encoder_input_to_vector(self, sentence: str) -> torch.Tensor:
        words = split_sentence_with_ko(sentence)
        length = self.sequence_size - 2

        if len(words) < length:
            words = [self.ko_voc.STR] + words + [self.ko_voc.END]
            words = words + [self.ko_voc.PAD for _ in range(self.sequence_size - len(words))]
        else:
            words = [self.ko_voc.STR] + words[:length] + [self.ko_voc.END]
        idx_list = self.word2idx(words, self.ko_voc)
        return torch.tensor(idx_list).to(device)

    def decoder_input_to_vector(self, sentence: str) -> torch.Tensor:
        words = split_sentence_with_en(sentence)
        words.insert(0, self.en_voc.STR)

        if len(words) < self.sequence_size:
            words = words + [self.en_voc.PAD for _ in range(self.sequence_size - len(words))]
        else:
            words = words[:self.sequence_size]
        idx_list = self.word2idx(words, self.en_voc)
        return torch.tensor(idx_list).to(device)

    def decoder_output_to_vector(self, sentence: str) -> torch.Tensor:
        words = split_sentence_with_en(sentence)
        words.append(self.en_voc.END)

        if len(words) < self.sequence_size:
            words = words + [self.en_voc.PAD for _ in range(self.sequence_size - len(words))]
        else:
            words = words[:self.sequence_size]
        idx_list = self.word2idx(words, self.en_voc)
        return torch.tensor(idx_list).to(device)

    @staticmethod
    def word2idx(words, voc):
        idx_list = []
        for word in words:
            try:
                idx_list.append(voc.word2idx[word])
            except KeyError:
                idx_list.append(voc.word2idx[voc.UNK])
        return idx_list


class RNNSeq2SeqDatasetV2(TranslationDataset):
    # for Seq2Seq model & Seq2Seq attention model
    # using google sentencepiece (https://github.com/google/sentencepiece.git)
    def __init__(self, x_path, y_path, ko_voc, en_voc, sequence_size):
        super().__init__(x_path, y_path, ko_voc, en_voc, sequence_size)
        self.KO_PAD_ID = ko_voc['<pad>']
        self.EN_PAD_ID = en_voc['<pad>']
        self.EN_BOS_ID = en_voc['<s>']
        self.EN_EOS_ID = en_voc['</s>']

    def __getitem__(self, idx):
        encoder_input = self.encoder_input_to_vector(self.x[idx])
        decoder_input = self.decoder_input_to_vector(self.y[idx])
        decoder_output = self.decoder_output_to_vector(self.y[idx])
        return encoder_input, decoder_input, decoder_output

    def encoder_input_to_vector(self, sentence: str):
        idx_list = self.ko_voc.EncodeAsIds(sentence)
        idx_list = self.padding(idx_list, self.KO_PAD_ID)
        return torch.tensor(idx_list).to(device)

    def decoder_input_to_vector(self, sentence: str):
        idx_list = self.en_voc.EncodeAsIds(sentence)
        idx_list.insert(0, self.EN_BOS_ID)
        idx_list = self.padding(idx_list, self.EN_PAD_ID)

        return torch.tensor(idx_list).to(device)

    def decoder_output_to_vector(self, sentence: str):
        idx_list = self.en_voc.EncodeAsIds(sentence)
        idx_list.append(self.EN_EOS_ID)
        idx_list = self.padding(idx_list, self.EN_PAD_ID)

        return torch.tensor(idx_list).to(device)

    def padding(self, idx_list, padding_id):
        length = len(idx_list)
        if length < self.sequence_size:
            idx_list = idx_list + [padding_id for _ in range(self.sequence_size - len(idx_list))]
        else:
            idx_list = idx_list[:self.sequence_size]
        return idx_list
