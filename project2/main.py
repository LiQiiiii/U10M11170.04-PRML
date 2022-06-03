# -*- coding: utf-8 -*-
# @Time    : 2022/6/2 19:59
# @Author  : Li Qi
# @FileName: main.py
# @Function: entrance for models

import os
import model_OnePath # 将一个数据集拆分成训练集和测试集
import model_TwoPath # 有单独的训练集和测试集

# 通过 os.environ 获取环境变量
os.environ['PATH'] = os.pathsep + r'C:\Users\61060\Anaconda3\pkgs\graphviz-2.38.0-4\Library\bin\graphviz'

# 将一个数据集拆分成训练集和测试集
path = "student-por.csv"
model_OnePath.knn_draft(path,feature_length=32) # 实现KNN，feature_length可选 30 或 32
model_OnePath.lg_draft(path,feature_length=32) # 实现Logistic Regression，feature_length可选 30 或 32
model_OnePath.dt_draft(path,feature_length=32) # 实现Decision Tree，feature_length可选 30 或 32

# 有单独的训练集和测试集
# trainset_path = ".csv"
# testset_path = ".csv"
# model_TwoPath.knn(trainset_path,testset_path,feature_length=32) # 实现KNN，feature_length可选 30 或 32
# model_TwoPath.logistic_regression(trainset_path,testset_path,feature_length=32) # 实现KNN，feature_length可选 30 或 32
# model_TwoPath.decision_tree(trainset_path,testset_path,feature_length=32) # 实现KNN，feature_length可选 30 或 32
