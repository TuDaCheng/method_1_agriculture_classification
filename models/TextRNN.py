# -*- coding: utf-8 -*-
# @Time : 2022/3/11 14:51
# @Author : TuDaCheng
# @File : TextRNN.py
import torch
import torch.nn as nn
import numpy as np


class Config(object):
    def __init__(self, embedding):
        self.model_name = "TextRNN"
        self.train_path = "./datas/train.txt"
        self.dev_path = "./datas/dev.txt"
        self.test_path = "./datas/test.txt"
        self.data_path = "./datas/agriculture_data.xlsx"
        self.vocab_path = "./datas/vocab_dict.pkl"
        self.dataset_pkl = "./datas/dataset_pkl"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.class_list = [x.strip() for x in open("datas/class.txt", encoding="utf-8").readlines()]
        self.save_path = "save_model/" + self.model_name + ".ckpt"
        self.padding_size = 20
        self.num_vocab = 0
        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_vocab = 0  # 词表大小 在运行时赋值
        self.num_epochs = 20
        self.learning_rate = 1e-3
        self.batch_size = 128
        self.embedding_pretrained = torch.tensor(
            np.load(embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300  # 字向量维度
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        content = x[0]
        content = torch.clamp(input=content, min=0, max=2362)
        out = self.embedding(content)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

