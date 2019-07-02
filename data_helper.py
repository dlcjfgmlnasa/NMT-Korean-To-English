# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import os
import re
import time
import pickle
from konlpy.tag import Okt


okt = Okt()


class Voc(object):
    def __init__(self, lng='ko'):
        self.lng = lng
        self.words = set()
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}

        if self.lng not in ['ko', 'en']:
            raise KeyError('voc supported only [ko, en]')

    def split_func(self):
        if self.lng == 'ko':
            return split_sentence_with_ko
        elif self.lng == 'en':
            return split_sentence_with_en

    def add_sentences(self, sentences):
        start_time = time.time()
        for sentence in sentences:
            self.words.update(self.split_func()(sentence))

            # make word2count
            for word in self.split_func()(sentence):
                if word in self.word2count:
                    self.word2count[word] += 1
                else:
                    self.word2count[word] = 1

        # make word2idx, idx2word
        words = sorted(list(self.words))
        for i, word in enumerate(words):
            self.word2idx[word] = i
            self.idx2word[i] = word

        end_time = time.time()
        print('load dictionary time : {:.3f}sec'.format(end_time - start_time))

    def trim(self, min_count=0, max_count=8000):
        trim_words = set()

        for word, count in self.word2count.items():
            if not (min_count <= count <= max_count):
                trim_words.add(word)
        self.words = self.words - trim_words

        # make word2idx, idx2word
        words = sorted(list(self.words))
        self.word2idx = {}
        self.idx2word = {}
        for i, word in enumerate(words):
            self.word2idx[word] = i
            self.idx2word[i] = word


def clean_str(string):
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


def create_or_get_voc(data_path, lng='ko', min_count=0, max_count=8000, save_path=None):
    if lng not in ['ko', 'en']:
        raise KeyError('voc supported only [ko, en]')

    lines = open(data_path, 'r', encoding='utf-8').readlines()
    file_path = os.path.join(save_path, lng + '.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            voc = pickle.load(f)
        return voc
    else:
        voc = None
        if lng == 'ko':
            voc = Voc(lng='ko')
        elif lng == 'en':
            voc = Voc(lng='en')

        voc.add_sentences(lines)
        voc.trim(min_count, max_count)

        with open(os.path.join(save_path, lng+'.pkl'), 'wb') as f:
            pickle.dump(voc, f, pickle.HIGHEST_PROTOCOL)
        return voc

