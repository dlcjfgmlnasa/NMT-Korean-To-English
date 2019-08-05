# NMT-Koean-To-English

작성중...

- 한영 기계 번역(Korean-English Machine Translation) 모델 개발 스터디 
- `PyTorch`, `koNLPY`, `NLPY`, `Gensim` package 활용 

## Requirements

- Python 3.6 (may work with other versions, but I used 3.6)
- PyTorch 1.1.0
- Gensim 3.8.0
- konlpy 0.5.1
- nltk 3.4.4

## Datasets
- https://github.com/jungyeul/korean-parallel-corpora 데이터셋 사용

```
git clone https://github.com/dlcjfgmlnasa/NMT-Koean-To-English.git --recursive
```

```
pip install -r requirement.txt
```

---

## 목차
1. [Sequence to Sequence (Seq2Seq)](#1.-Seq2Seq)
2. [Sequence to Sequence with Attention (Seq2Seq with Attention)](#2.-Seq2Seq-with-Attention)
3. [Convolution Sequence to Sequence](#3.-Convolution-Seq2Seq)
4. [ByteNet](#4.-ByteNet)
5. [SliceNet](#5.-SliceNet)
6. [Transformer](#6.-Transformer)


### 1. Seq2Seq
- Parameter List

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rnn_sequence_size', default=30, type=int)
parser.add_argument('--min_count', default=3, type=int)
parser.add_argument('--max_count', default=10000, type=int)
parser.add_argument('--embedding_size', default=200, type=int)
parser.add_argument('--rnn_dim', default=200, type=int)
parser.add_argument('--rnn_layer', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)
```

- loss

![Seq2Seq Loss function](./img/Seq2Seq_Loss_Graph.png)

- translation


### 2. Seq2Seq with Attention
- Parameter List
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rnn_sequence_size', default=30, type=int)
parser.add_argument('--min_count', default=3, type=int)
parser.add_argument('--max_count', default=100000, type=int)
parser.add_argument('--embedding_size', default=200, type=int)
parser.add_argument('--rnn_dim', default=123, type=int)
parser.add_argument('--rnn_layer', default=3, type=int)
parser.add_argument('--rnn_dropout_rate', default=0.5, type=float)
parser.add_argument('--use_residual', default=True, type=bool)
parser.add_argument('--attention_method', default='general', choices=['dot', 'general', 'concat'], type=str)
parser.add_argument('--batch_size', default=128, type=int)
```

### 3. Convolution Seq2Seq
- 구현 완료 (2019.08.05)


### 4. ByteNet


### 5. SliceNet


### 6. Transformer

## Reference
- https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- https://github.com/bentrevett/pytorch-seq2seq
- https://wikidocs.net/24996
