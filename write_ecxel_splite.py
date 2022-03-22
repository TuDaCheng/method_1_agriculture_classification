# -*- coding: utf-8 -*-
# @Time : 2022/3/4 16:11
# @Author : TuDaCheng
# @File : write_ecxel_splite.py

from sklearn.model_selection import train_test_split
import xlrd
import xlwt


def split_data(path):
    """
    将数据按6：2：2将数据分隔为训练集、测试集、验证集
    :param path:
    :return:
    """
    f1 = xlrd.open_workbook(path)
    sheet = f1.sheet_by_index(0)
    rows = sheet.nrows
    data = [[] for i in range(rows-1)]

    for i in range(1, rows):
        data[i-1] = sheet.row_values(i)[0:2]  # 去掉序号，取四个数据

    for i in range(0, len(data)):
        # 如果是String类型的数据，strip()方法那么可以将文本前后的所得空格去掉
        if isinstance(data[i][0], str):
            data[i][0] = data[i][0].strip()
    # 分为测试集和训练集，测试集占比0.2
    train, c_test = train_test_split(data, test_size=0.2, random_state=42)
    # 将训练集的四分之一设置为验证集，使训练集、测试集、验证集的比例为6：2：2
    c_train, c_dev = train_test_split(train, test_size=0.25, random_state=42)

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Sheet")

    for i in range(len(c_train)):
        for j in range(len(c_train[i])):
            sheet.write(i, j, c_train[i][j])
    workbook.save("datas/train.xls")

    workbook2 = xlwt.Workbook()
    sheet = workbook2.add_sheet("Sheet")

    for i in range(len(c_test)):
        for j in range(len(c_test[i])):
            sheet.write(i, j, c_test[i][j])
    workbook2.save("datas/test.xls")

    workbook3 = xlwt.Workbook()
    sheet = workbook3.add_sheet("Sheet")

    for i in range(len(c_dev)):
        for j in range(len(c_dev[i])):
            sheet.write(i, j, c_dev[i][j])
    workbook3.save("datas/dev.xls")


if __name__ == '__main__':
    data_path = "datas/agriculture_data.xls"
    split_data(data_path)
    pass
