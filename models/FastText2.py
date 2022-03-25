# -*- coding: utf-8 -*-
# @Time : 2022/3/11 14:22
# @Author : TuDaCheng
# @File : FastText2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """改进的fastText配置参数"""
    def __init__(self, embedding):
        self.model_name = "FastText2"
        self.train_path = "./datas/train.txt"
        self.dev_path = "./datas/dev.txt"
        self.test_path = "./datas/test.txt"
        self.data_path = "./datas/agriculture_data.xlsx"
        self.vocab_path = "./datas/vocab_dict2.pkl"
        self.dataset_pkl = "./datas/dataset_pkl2"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.class_list = [x.strip() for x in open("datas/class.txt", encoding="utf-8").readlines()]
        self.save_path = "save_model/" + self.model_name + ".ckpt"
        self.padding_size = 20
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
        self.hidden_size = 256                                          # 隐藏层大小
        self.n_gram_vocab = 250499                                      # ngram 词表大小


'''Bag of Tricks for Efficient Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.num_vocab, config.embed, padding_idx=config.num_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        content = x[0]
        content = torch.clamp(input=content, min=0, max=2362)
        out_word = self.embedding(content)
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out