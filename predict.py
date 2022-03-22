# -*- coding: utf-8 -*-
# @Time : 2022/3/10 15:41
# @Author : TuDaCheng
# @File : predict.py

import torch
import time
import argparse
import data_utils
import numpy as np
from importlib import import_module
from train import predict


# 参数配置
parser = argparse.ArgumentParser(description="Chinese Text Classification")  # 声明argparse对象 可附加说明
# 添加模型参数 模型是必须设置的参数(required=True) 类型是字符串
parser.add_argument("--model", type=str, default="TextCNN", help="choose a model: TextCNN, TextRNN")
# embedding随机初始化或使用预训练词或字向量 默认使用预训练
parser.add_argument("--embedding", default="pre_trained", type=str, help="random or pre_trained")
# 基于词还是基于字 默认基于字
parser.add_argument("--word", default=False, type=bool, help="True for word, False for char")

# 解析参数
args = parser.parse_args()

if __name__ == '__main__':

    model_name = args.model
    x = import_module("models." + model_name)  # 根据所选模型名字在models包下 获取相应模块(.py)
    embedding = "datas/embedding.npz"
    config = x.Config(embedding)
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load('./save_model/' + 'TextCNN.ckpt'))
    # text = input("输入预测的句子:")
    text = "今年的鱼虾养殖不是很好的行业"
    predict(model, text=text)

