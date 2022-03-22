# -*- coding: utf-8 -*-
# @Time : 2022/3/9 16:55
# @Author : TuDaCheng
# @File : train.py
import pickle
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import data_utils
from data_utils import get_time_dif
import torch.nn.functional as F
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    f_writer = open("./result_data/train_result_data_" + config.model_name, "w")
    f_writer.write(
        "Iter" + "\t" + " Train Loss" + "\t" + "Train Acc" +
        "\t"  + "Val Loss" + "\t" + "Val Acc" + "\t" + "Time" + "\n")
    total_batch = 0
    dev_best_loss = float("inf")
    last_improve = 0
    flag = False
    for epoch in range(config.num_epochs):
        # print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 5 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

                f_writer.write(str(total_batch) + "\t" + str(loss.item()) + "\t" +
                               str(train_acc) + "\t" + str(dev_loss) + "\t"
                               + str(dev_acc) + "\t" + str(time_dif) + "\n")
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = "Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision Recall and F1-score")
    print(test_report)
    print("Confusion Matrix")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time Useg:", time_dif)


def predict(model, text):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_dict = {1: "农作物", 2: "园艺", 3: "林业", 4: "养殖技术", 5: "水产渔业", 6: "农业工程", 7: "农业经济", 8: "农业法规"}
    with torch.no_grad():
        "将输入的句子转化成id"
        text, _ = data_utils.preprocessing_text(text)
        word_to_id = pickle.load(open("datas/vocab_dict.pkl", "rb"))
        word_list = []
        idx_list = []

        input_tuple = []

        for word in text:
            word_list.append(word)
        if len(word_list) < 20:
            word_list.extend(["PAD"] * (20 - len(word_list)))
        else:
            word_list = word_list[:20]

        for char in word_list:
            if char in word_to_id:
                idx = word_to_id[char]
            else:
                idx = word_to_id["UNK"]
            idx_list.append(idx)
        idx_list = torch.LongTensor(idx_list).to(device)
        idx_list = idx_list.unsqueeze(0)
        print(idx_list)
        input_tuple.append(idx_list)
        input_tuple.append(word_list)
        input_tuple = tuple(input_tuple)

        output = model(input_tuple)
        probability = torch.nn.functional.softmax(output.data, dim=1)
        probability = np.round(probability.cpu().detach().numpy(), 3)
        predic_label_id = torch.max(output.data, 1)[1].cpu().numpy()
        predict = label_dict[int(predic_label_id)]
        print(predict)


if __name__ == '__main__':
    train()