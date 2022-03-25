# -*- coding: utf-8 -*-
# @Time : 2022/3/22 18:47
# @Author : TuDaCheng
# @File : FastText1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, embedding):
        self.model_name = "FastText1"
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


class Model(nn.Module):  # 继承自BasicModule 其中封装了保存加载模型的接口,BasicModule继承自nn.Module
    def __init__(self, config):
        super(Model, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(config.num_vocab, config.embed, padding_idx=config.num_vocab - 1)  # 词嵌入矩阵 每一行代表词典中一个词对应的词向量；
        # 词嵌入矩阵可以随机初始化连同分类任务一起训练，也可以用预训练词向量初始化（冻结或微调）

        self.content_fc = nn.Sequential(  # 可以使用多个全连接层或batchnorm、dropout等 可以把这些模块用Sequential包装成一个大模块
            nn.Linear(config.embed, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(inplace=True),
            # 可以再加一个隐层
            # nn.Linear(opt.linear_hidden_size,opt.linear_hidden_size),
            # nn.BatchNorm1d(opt.linear_hidden_size),
            # nn.ReLU(inplace=True),
            # 输出层
            nn.Linear(config.hidden_size, config.num_classes)
        )

    def forward(self, x):
        # inputs(batch_size,seq_len)
        content = x[0]
        # content = torch.clamp(input=content, min=0, max=2362)
        out = self.embedding(content)  # (batch_size, seq_len, embed_size)

        # 对seq_len维取平均
        out = torch.mean(out, dim=1, keepdim=True)  # (batch_size,1,embed_size)


        out = self.content_fc(out.squeeze(1))  # 先压缩seq_len维 (batch_size,embed_size) 然后作为全连接层的输入
        # 输出 (batch_size,classes)

        return out



