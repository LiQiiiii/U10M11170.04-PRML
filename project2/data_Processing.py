# -*- coding: utf-8 -*-
# @Time    : 2022/6/2 19:53
# @Author  : Li Qi
# @FileName: data_Processing.py
# @Function: pre-processing for data

from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import math

def data_preprocessing_30(path):
    ##### 数据处理 #####
    data_original = pd.read_csv(path, sep=';')
    X_original = data_original.iloc[:, :30]

    X_array = np.array(X_original)
    data_array = np.array(data_original)

    X_list = X_array.tolist()
    data_list = data_array.tolist()
    # 将字符串型数据转换为数字型数据
    enc = OrdinalEncoder()
    enc.fit(X_list)
    X = enc.transform(X_list)
    G1 = data_array[:, 30]
    G2 = data_array[:, 31]
    G3 = data_array[:, 32]
    X = X.astype('int')
    G1 = G1.astype('int')
    G2 = G2.astype('int')
    G3 = G3.astype('int')
    # 将成绩分成五个等级
    ############## G1 ###############
    print("G1 preprocessing begin!")
    y_sort = np.sort(G1)
    y_rank1 = y_sort[:math.floor(len(y_sort) * 0.1)]
    y_rank2 = y_sort[math.floor(len(y_sort) * 0.1):math.floor(len(y_sort) * 0.3)]
    y_rank3 = y_sort[math.floor(len(y_sort) * 0.3):math.floor(len(y_sort) * 0.7)]
    y_rank4 = y_sort[math.floor(len(y_sort) * 0.7):math.floor(len(y_sort) * 0.9)]
    y_rank5 = y_sort[math.floor(len(y_sort) * 0.9):math.floor(len(y_sort))]

    for i in range(len(G1)):
        if (G1[i] >= y_rank5[0] and G1[i] <= y_rank5[len(y_rank5) - 1]):
            G1[i] = 5
        elif (G1[i] >= y_rank4[0] and G1[i] <= y_rank4[len(y_rank4) - 1]):
            G1[i] = 4
        elif (G1[i] >= y_rank3[0] and G1[i] <= y_rank3[len(y_rank3) - 1]):
            G1[i] = 3
        elif (G1[i] >= y_rank2[0] and G1[i] <= y_rank2[len(y_rank2) - 1]):
            G1[i] = 2
        elif (G1[i] >= y_rank1[0] and G1[i] <= y_rank1[len(y_rank1) - 1]):
            G1[i] = 1
        else:
            G1[i] = 0  # 0代表不存在该分数，预测错误。
    print("G1 preprocessing done!")

    ############## G2 ###############
    print("G2 preprocessing begin!")
    y_sort = np.sort(G2)
    y_rank1 = y_sort[:math.floor(len(y_sort) * 0.1)]
    y_rank2 = y_sort[math.floor(len(y_sort) * 0.1):math.floor(len(y_sort) * 0.3)]
    y_rank3 = y_sort[math.floor(len(y_sort) * 0.3):math.floor(len(y_sort) * 0.7)]
    y_rank4 = y_sort[math.floor(len(y_sort) * 0.7):math.floor(len(y_sort) * 0.9)]
    y_rank5 = y_sort[math.floor(len(y_sort) * 0.9):math.floor(len(y_sort))]

    for i in range(len(G2)):
        if (G2[i] >= y_rank5[0] and G2[i] <= y_rank5[len(y_rank5) - 1]):
            G2[i] = 5
        elif (G2[i] >= y_rank4[0] and G2[i] <= y_rank4[len(y_rank4) - 1]):
            G2[i] = 4
        elif (G2[i] >= y_rank3[0] and G2[i] <= y_rank3[len(y_rank3) - 1]):
            G2[i] = 3
        elif (G2[i] >= y_rank2[0] and G2[i] <= y_rank2[len(y_rank2) - 1]):
            G2[i] = 2
        elif (G2[i] >= y_rank1[0] and G2[i] <= y_rank1[len(y_rank1) - 1]):
            G2[i] = 1
        else:
            G2[i] = 0  # 0代表不存在该分数，预测错误。
    print("G2 preprocessing done!")

    ############## G3 ###############
    print("G3 preprocessing begin!")
    y_sort = np.sort(G3)
    y_rank1 = y_sort[:math.floor(len(y_sort) * 0.1)]
    y_rank2 = y_sort[math.floor(len(y_sort) * 0.1):math.floor(len(y_sort) * 0.3)]
    y_rank3 = y_sort[math.floor(len(y_sort) * 0.3):math.floor(len(y_sort) * 0.7)]
    y_rank4 = y_sort[math.floor(len(y_sort) * 0.7):math.floor(len(y_sort) * 0.9)]
    y_rank5 = y_sort[math.floor(len(y_sort) * 0.9):math.floor(len(y_sort))]

    for i in range(len(G3)):
        if (G3[i] >= y_rank5[0] and G3[i] <= y_rank5[len(y_rank5) - 1]):
            G3[i] = 5
        elif (G3[i] >= y_rank4[0] and G3[i] <= y_rank4[len(y_rank4) - 1]):
            G3[i] = 4
        elif (G3[i] >= y_rank3[0] and G3[i] <= y_rank3[len(y_rank3) - 1]):
            G3[i] = 3
        elif (G3[i] >= y_rank2[0] and G3[i] <= y_rank2[len(y_rank2) - 1]):
            G3[i] = 2
        elif (G3[i] >= y_rank1[0] and G3[i] <= y_rank1[len(y_rank1) - 1]):
            G3[i] = 1
        else:
            G3[i] = 0  # 0代表不存在该分数，预测错误。
    print("G3 preprocessing done!")
    # print("final X: ")
    # print(X)
    # print("final y: ")
    # print(y)
    return X, G1, G2, G3

def data_preprocessing_32(path):
    ##### 数据处理 #####
    data_original = pd.read_csv(path, sep=';')
    X_original = data_original.iloc[:, :32]

    X_array = np.array(X_original)
    data_array = np.array(data_original)

    X_list = X_array.tolist()
    data_list = data_array.tolist()
    # 将字符串型数据转换为数字型数据
    enc = OrdinalEncoder()
    enc.fit(X_list)
    X = enc.transform(X_list)
    y = data_array[:, 32]
    X = X.astype('int')
    y = y.astype('int')
    # 将成绩分成五个等级
    y_sort = np.sort(y)
    y_rank1 = y_sort[:math.floor(len(y_sort) * 0.1)]
    y_rank2 = y_sort[math.floor(len(y_sort) * 0.1):math.floor(len(y_sort) * 0.3)]
    y_rank3 = y_sort[math.floor(len(y_sort) * 0.3):math.floor(len(y_sort) * 0.7)]
    y_rank4 = y_sort[math.floor(len(y_sort) * 0.7):math.floor(len(y_sort) * 0.9)]
    y_rank5 = y_sort[math.floor(len(y_sort) * 0.9):math.floor(len(y_sort))]

    for i in range(len(y)):
        if (y[i] >= y_rank5[0] and y[i] <= y_rank5[len(y_rank5) - 1]):
            y[i] = 5
        elif (y[i] >= y_rank4[0] and y[i] <= y_rank4[len(y_rank4) - 1]):
            y[i] = 4
        elif (y[i] >= y_rank3[0] and y[i] <= y_rank3[len(y_rank3) - 1]):
            y[i] = 3
        elif (y[i] >= y_rank2[0] and y[i] <= y_rank2[len(y_rank2) - 1]):
            y[i] = 2
        elif (y[i] >= y_rank1[0] and y[i] <= y_rank1[len(y_rank1) - 1]):
            y[i] = 1
        else:
            y[i] = 0  # 0代表不存在该分数，预测错误。
    # print("final X: ")
    # print(X)
    # print("final y: ")
    # print(y)
    return X,y

