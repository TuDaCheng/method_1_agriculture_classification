# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


UNK, PAD = 'UNK', 'PAD'


def build_vocab(file_path):
    vocab_dict = {}
    max_size = 10000
    min_freq = 1
    tokenizer = lambda x: [y for y in x]
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            lens = len(line.strip().split("\t"))
            if lens == 3:
                _, content, label = lin.split('\t')
                for word in tokenizer(content):
                    vocab_dict[word] = vocab_dict.get(word, 0) + 1

                    # 过来低频词排序 取出max_size个单词
                    vocab_list = sorted([item for item in vocab_dict.items() if item[1] >= min_freq],
                                        key=lambda x: x[1], reverse=True)[:max_size]
                    # 构建字典映射
                    vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
                    vocab_dict.update({UNK: len(vocab_dict), PAD: len(vocab_dict) + 1})

    with open("datas/vocab_dict2.pkl", "wb") as f_writer:
        pkl.dump(vocab_dict, f_writer)
    return vocab_dict


def get_dict(path):
    """
    加载字典
    :param path:
    :return:
    """
    with open(path, "rb") as f_reader:
        vocab_dict = pkl.load(f_reader)
    return vocab_dict


def load_dataset(path, config):
    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets
    contents = []
    # 对句子进行分字处理
    tokenizer = lambda s: [w for w in s]
    vocab = get_dict("datas/vocab_dict2.pkl")
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lens = len(line.strip().split("\t"))
            if lens == 2:
                content, label = line.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if token:
                    if len(token) < config.padding_size:
                        token.extend([PAD] * (config.padding_size - len(token)))
                    else:
                        token = token[:config.padding_size]
                        seq_len = config.padding_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                # fasttext ngram
                buckets = config.n_gram_vocab
                bigram = []
                trigram = []
                # ------ngram------
                for i in range(config.padding_size):
                    bigram.append(biGramHash(words_line, i, buckets))
                    trigram.append(triGramHash(words_line, i, buckets))
                # -----------------
                contents.append((words_line, int(label), seq_len, bigram, trigram))
    return contents  # [([...], 0), ([...], 1), ...]


def build_dataset(config):

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    if os.path.exists(config.dataset_pkl):

        dataset = pkl.load(open(config.dataset_pkl, "rb"))
        # print(dataset.get("train_data"))
        train_data = dataset["train_data"]
        # print(len(train_data))
        dev_data = dataset["dev_data"]
        test_data = dataset["test_data"]
    else:
        # 分别对 训练集 验证集 测试集进行处理 把文本中的词转化成字典中的索引id
        train_data = load_dataset(config.train_path, config)
        dev_data = load_dataset(config.dev_path, config)
        test_data = load_dataset(config.test_path, config)
        dataset = {}
        dataset["train_data"] = train_data
        dataset["dev_data"] = dev_data
        dataset["test_data"] = test_data
        pkl.dump(dataset, open(config.dataset_pkl, "wb"))
    return vocab, train_data, dev_data, test_data


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches  # 构建好的数据集
        self.n_batches = len(batches) // batch_size  # 得到batch数量
        # print(len(batches))
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:  # 不能整除
            self.residue = True  # True表示不能整除
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # xx = [xxx[2] for xxx in datas]
        # indexx = np.argsort(xx)[::-1]
        # datas = np.array(datas)[indexx]
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    path = "datas/agriculture_data.txt"
    vocab_dic = build_vocab(path)
    # with open("datas/vocab_dict2.pkl", "rb") as f_reader:
    #     data = pkl.load(f_reader)
    # print(len(data))
    # word = "什"
    # print(data.get(word, data.get(UNK)))
    # # print(data)

    # text = "11111aaaaaa你好啊，?★、…【】《》？"
    # con, stop_word_list = preprocessing_text(text)

    # config = config
    # build_dataset(config)
