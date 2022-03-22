# -*- coding: utf-8 -*-
# @Time : 2022/3/11 15:09
# @Author : TuDaCheng
# @File : TextDPCNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    def __init__(self, embedding):
        self.model_name = "TextDPCNN"
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
        self.filter_size = (2, 3, 4)
        self.num_filters = 256                                    # 卷积核数量(channels数)


'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:  # 加载初始化好的预训练词/字嵌入矩阵  微调funetuning
            print(config.embedding_pretrained.size())
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            print(self.embedding)
        else:  # 否则随机初始化词/字嵌入矩阵 指定填充对应的索引
            self.embedding = nn.Embedding(config.num_vocab, config.embed, padding_idx=config.num_vocab - 1)

        # region embedding 类似于TextCNN中的卷积操作
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)

        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()

        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def _block(self, x):
        out = self.padding2(x)  # [batch_size, num_filters, seq_len-1, 1]
        # 长度减半
        p_out = self.max_pool(out)  # [batch_size, num_filters, （seq_len-1）/2, 1]

        # 等长卷积 长度不变
        out = self.padding1(p_out)
        out = F.relu(out)
        out = self.conv(out)

        # 等长卷积 长度不变
        out = self.padding1(out)
        out = F.relu(out)
        out = self.conv(out)

        # short Cut
        out = out + p_out
        return out

    def forward(self, x):
        content = x[0]  # [batch_size, seq_length]
        content = torch.clamp(input=content, min=0, max=2362)
        out = self.embedding(content)  # [batch_size, seq_length, embedding_dim]
        out = out.unsqueeze(1)  # 添加通道维 进行2d卷积[batch_size, 1, seq_length, embedding_dim]
        out = self.conv_region(out)  # (batch_size,num_filters,seq_length-3+1,1)
        # 先卷积 再填充 等价于等长卷积  序列长度不变
        out = self.padding1(out)  # [batch_size, num_filters, seq_len, 1]
        out = self.relu(out)

        out = self.conv(out)  # [batch_size, num_filters, seq_len-3+1, 1]
        out = self.padding1(out)  # [batch_size, num_filters, seq_len, 1]
        out = self.relu(out)

        out = self.conv(out)  # [batch_size, num_filters, seq_len-3+1, 1]
        while out.size()[2] > 1:
            out = self._block(out)
        out = out.squeeze()
        out = self.fc(out)

        return out