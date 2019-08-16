# -*- coding:utf-8 -*-
import argparse
from tools import Trainer, Translation


def get_args():
    parser = argparse.ArgumentParser()
    # file path
    parser.add_argument('--data_path', default='../Dataset', type=str)
    parser.add_argument('--dictionary_path', default='../Dictionary', type=str)
    parser.add_argument('--src_train_filename', default='train.ko', type=str)
    parser.add_argument('--trg_train_filename', default='train.en', type=str)
    parser.add_argument('--src_val_filename', default='dev.ko', type=str)
    parser.add_argument('--trg_val_filename', default='dev.en', type=str)
    parser.add_argument('--model_path', default='save_model/', type=str)
    parser.add_argument('--img_path', default='img/', type=str)

    # Model Hyper Parameter
    parser.add_argument('--sequence_size', default=40, type=int)
    parser.add_argument('--embedding_dim', default=200, type=int)

    # 1. Decoder
    parser.add_argument('--encoder_rnn_dim', default=100, type=int)
    parser.add_argument('--encoder_n_layers', default=3, type=int)
    parser.add_argument('--encoder_embedding_dropout', default=0.5, type=float)
    parser.add_argument('--encoder_rnn_dropout', default=0.5, type=float)
    parser.add_argument('--encoder_dropout', default=0.5, type=float)
    parser.add_argument('--encoder_bidirectional_used', default=True, type=float)
    parser.add_argument('--encoder_residual_used', default=True, type=bool)
    parser.add_argument('--encoder_output_transformer', default=100)
    parser.add_argument('--encoder_output_transformer_bias', default=True, type=bool)
    parser.add_argument('--encoder_hidden_transformer', default=100)
    parser.add_argument('--encoder_hidden_transformer_bias', default=True, type=bool)
    parser.add_argument('--encoder_weight_norm_used', default=True, type=bool)

    # 2. Decoder
    parser.add_argument('--decoder_rnn_dim', default=100, type=int)
    parser.add_argument('--decoder_n_layers', default=3, type=int)
    parser.add_argument('--decoder_embedding_dropout', default=0.5, type=float)
    parser.add_argument('--decoder_dropout', default=0.5, type=float)
    parser.add_argument('--decoder_rnn_dropout', default=0.5, type=float)
    parser.add_argument('--decoder_residual_used', default=True, type=bool)
    parser.add_argument('--decoder_weight_norm_used', default=True, type=bool)

    # 3. Attention
    parser.add_argument('--attention_score', default='general', type=str, choices=['dot', 'general', 'concat'])
    parser.add_argument('--get_attention', default=True, type=bool)

    # 3. learning hyper parameter
    parser.add_argument('--learning_method', default='Scheduled_Sampling', type=str,
                        choices=['Teacher_Forcing', 'Scheduled_Sampling'])
    parser.add_argument('--learning_rate', default=0.005, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--plot_count', default=6, type=int)
    parser.add_argument('--train_step_print', default=10, type=int)
    parser.add_argument('--val_step_print', default=100, type=int)
    parser.add_argument('--step_save', default=500, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    Trainer(args)

    # translation = Translation(
    #     checkpoint='./save_model/attention_01_seq2seq.pth',
    #     dictionary_path='../Dictionary',
    #     get_attention=False
    # )
    # test = translation.translation('슈월제네거 주지사는 캘리포니아주 예산 위기에 대해 언급하며 공공 안전이 최우선 사항임을 강조했다.')
    # print(test)
