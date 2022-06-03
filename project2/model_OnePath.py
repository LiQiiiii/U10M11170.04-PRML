# -*- coding: utf-8 -*-
# @Time    : 2022/6/2 19:56
# @Author  : Li Qi
# @FileName: model_OnePath.py
# @Function: three models with one input path

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from data_Processing import data_preprocessing_30
from data_Processing import data_preprocessing_32

####### 32-knn-接口1个数据集 #######
def knn_draft(path, feature_length):
    print("######################### KNN #########################")
    # 获得特征标签数据
    if(feature_length == 32):
        X, y = data_preprocessing_32(path)
        # 用最大似然估计自选新特征个数
        pca = PCA(n_components="mle")  # 实例化
        pca = pca.fit(X)  # 拟合模型
        X_dr = pca.transform(X)  # 获取新的特征
        print("Length of newly generated features: ")
        print(len(X_dr[0]))
        print("Explained variance: ")
        print(pca.explained_variance_)  # 查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
        print("Sum of explained variance ratio: ")
        print(pca.explained_variance_ratio_.sum())  # 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比之和，一般新特征能够反映原来特征的大部分信息（80%以上）即可

        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(X_dr, y, test_size=0.1)

        ##### 拟合模型 #####
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(x_train, y_train)
        # y_pre = knn.predict(x_test)
        # print("y_pre: ")
        # print(y_pre)
        # print("y_true: ")
        # print(y_test)

        ## 以下的注释是先用0~20的分数进行训练，然后将结果分等级。
        # y_pre_sort = np.sort(y_pre)
        # pre_rank1 = y_pre_sort[:math.floor(len(y_pre_sort)*0.1)]
        # pre_rank2 = y_pre_sort[math.floor(len(y_pre_sort)*0.1):math.floor(len(y_pre_sort)*0.3)]
        # pre_rank3 = y_pre_sort[math.floor(len(y_pre_sort)*0.3):math.floor(len(y_pre_sort)*0.7)]
        # pre_rank4 = y_pre_sort[math.floor(len(y_pre_sort)*0.7):math.floor(len(y_pre_sort)*0.9)]
        # pre_rank5 = y_pre_sort[math.floor(len(y_pre_sort)*0.9):math.floor(len(y_pre_sort))]
        # # print(pre_rank1)
        # # print(pre_rank2)
        # # print(pre_rank3)
        # # print(pre_rank4)
        # # print(pre_rank5)
        #
        # y_test_sort = np.sort(y_test)
        # test_rank1 = y_test_sort[:math.floor(len(y_test_sort)*0.1)]
        # test_rank2 = y_test_sort[math.floor(len(y_test_sort)*0.1):math.floor(len(y_test_sort)*0.3)]
        # test_rank3 = y_test_sort[math.floor(len(y_test_sort)*0.3):math.floor(len(y_test_sort)*0.7)]
        # test_rank4 = y_test_sort[math.floor(len(y_test_sort)*0.7):math.floor(len(y_test_sort)*0.9)]
        # test_rank5 = y_test_sort[math.floor(len(y_test_sort)*0.9):len(y_test_sort)]
        # # print(test_rank1)
        # # print(test_rank2)
        # # print(test_rank3)
        # # print(test_rank4)
        # # print(test_rank5)
        #
        #
        # for i in range(len(y_test)):
        #     if (y_test[i]>=test_rank5[0] and y_test[i]<=test_rank5[len(test_rank5)-1]):
        #         y_test[i] = 5
        #     elif(y_test[i]>=test_rank4[0] and y_test[i]<=test_rank4[len(test_rank4)-1]):
        #         y_test[i] = 4
        #     elif(y_test[i]>=test_rank3[0] and y_test[i]<=test_rank3[len(test_rank3)-1]):
        #         y_test[i] = 3
        #     elif(y_test[i]>=test_rank2[0] and y_test[i]<=test_rank2[len(test_rank2)-1]):
        #         y_test[i] = 2
        #     elif(y_test[i]>=test_rank1[0] and y_test[i]<=test_rank1[len(test_rank1)-1]):
        #         y_test[i] = 1
        #     else:
        #         y_test[i] = 0 # 0代表不存在该分数，预测错误。
        # print(y_test)
        #
        # for i in range(len(y_pre)):
        #     if (y_pre[i]>=pre_rank5[0] and y_pre[i]<=pre_rank5[len(pre_rank5)-1]):
        #         y_pre[i] = 5
        #     elif(y_pre[i]>=pre_rank4[0] and y_pre[i]<=pre_rank4[len(pre_rank4)-1]):
        #         y_pre[i] = 4
        #     elif(y_pre[i]>=pre_rank3[0] and y_pre[i]<=pre_rank3[len(pre_rank3)-1]):
        #         y_pre[i] = 3
        #     elif(y_pre[i]>=pre_rank2[0] and y_pre[i]<=pre_rank2[len(pre_rank2)-1]):
        #         y_pre[i] = 2
        #     elif(y_pre[i]>=pre_rank1[0] and y_pre[i]<=pre_rank1[len(pre_rank1)-1]):
        #         y_pre[i] = 1
        #     else:
        #         y_pre[i] = 0 # 0代表不存在该分数，预测错误。
        # print(y_pre)

        # correct = 0
        # for i in range(len(y_test)):
        #     if (y_pre[i] == y_test[i]):
        #         correct += 1
        # ratio = correct / len(y_test)
        # print("Accuracy: ")
        # print(ratio)
        Accuracy = accuracy_score(knn.predict(x_train), y_train)
        print('Accuracy: ')
        print(Accuracy)
    elif(feature_length == 30):
        X, G1, G2, G3 = data_preprocessing_30(path)
        # 用最大似然估计自选新特征个数
        pca = PCA(n_components="mle")  # 实例化
        pca = pca.fit(X)  # 拟合模型
        X_dr = pca.transform(X)  # 获取新的特征
        print("Length of newly generated features: ")
        print(len(X_dr[0]))
        print("Explained variance: ")
        print(pca.explained_variance_)  # 查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
        print("Sum of explained variance ratio: ")
        print(pca.explained_variance_ratio_.sum())  # 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比之和，一般新特征能够反映原来特征的大部分信息（80%以上）即可

        # 划分训练集和测试集
        x_train, x_test, g1_train, g1_test = train_test_split(X_dr, G1, test_size=0.1)
        x_train, x_test, g2_train, g2_test = train_test_split(X_dr, G2, test_size=0.1)
        x_train, x_test, g3_train, g3_test = train_test_split(X_dr, G3, test_size=0.1)
        ##### 拟合模型 #####
        knn_g1 = KNeighborsClassifier(n_neighbors=7)
        knn_g1.fit(x_train, g1_train)

        knn_g2 = KNeighborsClassifier(n_neighbors=7)
        knn_g2.fit(x_train, g2_train)

        knn_g3 = KNeighborsClassifier(n_neighbors=7)
        knn_g3.fit(x_train, g3_train)
        # y_pre = knn.predict(x_test)
        # print("y_pre: ")
        # print(y_pre)
        # print("y_true: ")
        # print(y_test)

        Accuracy_g1 = accuracy_score(knn_g1.predict(x_train), g1_train)
        Accuracy_g2 = accuracy_score(knn_g2.predict(x_train), g2_train)
        Accuracy_g3 = accuracy_score(knn_g3.predict(x_train), g3_train)
        print('Accuracy_G1: ')
        print(Accuracy_g1)
        print('Accuracy_G2: ')
        print(Accuracy_g2)
        print('Accuracy_G3: ')
        print(Accuracy_g3)
    else:
        print("Wrong Feature Length!!!")
        return 0
####### 32--logistic_regression--接口1个数据集 #######
def lg_draft(path,feature_length):
    print("######################### Logistic Regression #########################")
    # 获得特征标签数据
    if(feature_length == 32):
        X, y = data_preprocessing_32(path)
        # PCA降维
        pca = PCA(n_components=2)  # 实例化
        pca = pca.fit(X)  # 拟合模型
        X_dr = pca.transform(X)  # 获取新的特征
        print("Length of newly generated features: ")
        print(len(X_dr[0]))
        print("Explained variance: ")
        print(pca.explained_variance_)  # 查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
        print("Sum of explained variance ratio: ")
        print(pca.explained_variance_ratio_.sum())  # 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比之和

        # 分割训练测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

        lr_l1 = LogisticRegression(multi_class="ovr",penalty="l1", C=0.1, solver="liblinear")
        lr_l2 = LogisticRegression(multi_class="multinomial",penalty="l2", C=0.05, solver="newton-cg")

        # 训练模型
        lr_l1.fit(X_train, y_train)
        lr_l2.fit(X_train, y_train)

        Accuracy = accuracy_score(lr_l1.predict(X_test), y_test)
        print("Accuracy: ")
        print(Accuracy)

        ## 测试不同c值下模型的表现情况
        # 训练集表现
        l1_train_predict = []
        l2_train_predict = []
        # 测试集表现
        l1_test_predict = []
        l2_test_predict = []

        for c in np.linspace(0.01, 2, 50):
            lr_l1 = LogisticRegression(penalty="l1", C=c, solver="liblinear", max_iter=1000)
            lr_l2 = LogisticRegression(penalty='l2', C=c, solver='liblinear', max_iter=1000)
            # 训练模型，记录L1正则化模型在训练集测试集上的表现
            lr_l1.fit(X_train, y_train)
            l1_train_predict.append(accuracy_score(lr_l1.predict(X_train), y_train))
            l1_test_predict.append(accuracy_score(lr_l1.predict(X_test), y_test))
            # 记录L2正则化模型的表现
            lr_l2.fit(X_train, y_train)
            l2_train_predict.append(accuracy_score(lr_l2.predict(X_train), y_train))
            l2_test_predict.append(accuracy_score(lr_l2.predict(X_test), y_test))

        data = [l1_train_predict, l2_train_predict, l1_test_predict, l2_test_predict]
        label = ['l1_train', 'l2_train', 'l1_test', "l2_test"]
        color = ['maroon', 'turquoise', 'dodgerblue', 'slategray']

        plt.rc('font', family='Times New Roman')
        plt.figure(figsize=(5, 4))
        for i in range(4):
            plt.plot(np.linspace(0.01, 2, 50), data[i], label=label[i], color=color[i], linewidth=2.5)
        plt.title('Accuracy with different C value', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.xlabel('C value', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.ylabel('Accuracy', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.legend(loc="best")
        plt.show()
    elif(feature_length == 30):
        X, G1, G2, G3 = data_preprocessing_30(path)
        # PCA降维
        pca = PCA(n_components="mle")  # 实例化
        pca = pca.fit(X)  # 拟合模型
        X_dr = pca.transform(X)  # 获取新的特征
        print("Length of newly generated features: ")
        print(len(X_dr[0]))
        print("Explained variance: ")
        print(pca.explained_variance_)  # 查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
        print("Sum of explained variance ratio: ")
        print(pca.explained_variance_ratio_.sum())  # 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比之和

        # 分割训练测试集
        x_train, x_test, g1_train, g1_test = train_test_split(X_dr, G1, test_size=0.1)
        x_train, x_test, g2_train, g2_test = train_test_split(X_dr, G2, test_size=0.1)
        x_train, x_test, g3_train, g3_test = train_test_split(X_dr, G3, test_size=0.1)

        print("G1 model training and testing begin!")
        lr_l1 = LogisticRegression(multi_class="ovr",penalty="l1", C=1, solver="liblinear")
        # lr_l2 = LogisticRegression(penalty="l2", C=0.5, solver="liblinear")
        # 训练模型
        lr_l1.fit(x_train, g1_train)
        Accuracy_g1 = accuracy_score(lr_l1.predict(x_test), g1_test)
        print("Accuracy_G1: ")
        print(Accuracy_g1)
        print("G1 done!")

        print("G2 model training and testing begin!")
        lr_l1 = LogisticRegression(multi_class="ovr",penalty="l1", C=1, solver="liblinear")
        # lr_l2 = LogisticRegression(multi_class="multinomial",penalty="l2", C=0.5, solver="liblinear")
        # 训练模型
        lr_l1.fit(x_train, g2_train)
        Accuracy_g2 = accuracy_score(lr_l1.predict(x_test), g2_test)
        print("Accuracy_G2: ")
        print(Accuracy_g2)
        print("G2 done!")

        print("G3 model training and testing begin!")
        lr_l1 = LogisticRegression(multi_class="ovr",penalty="l1", C=1, solver="liblinear")
        # lr_l2 = LogisticRegression(penalty="l2", C=0.5, solver="liblinear")
        # 训练模型
        lr_l1.fit(x_train, g1_train)
        Accuracy_g1 = accuracy_score(lr_l1.predict(x_test), g1_test)
        print("Accuracy_G3: ")
        print(Accuracy_g1)
        print("G3 done!")

        ################## 测试不同c值下模型的表现情况: G1 ##################
        # 训练集表现
        l1_train_predict = []
        l2_train_predict = []
        # 测试集表现
        l1_test_predict = []
        l2_test_predict = []

        for c in np.linspace(0.01, 2, 50):
            lr_l1 = LogisticRegression(multi_class="ovr",penalty="l1", C=c, solver="liblinear", max_iter=1000)
            lr_l2 = LogisticRegression(multi_class="multinomial",penalty='l2', C=c, solver='newton-cg', max_iter=1000)

            # 训练模型，记录L1正则化模型在训练集测试集上的表现
            lr_l1.fit(x_train, g1_train)
            l1_train_predict.append(accuracy_score(lr_l1.predict(x_train), g1_train))
            l1_test_predict.append(accuracy_score(lr_l1.predict(x_test), g1_test))

            # 记录L2正则化模型的表现
            lr_l2.fit(x_train, g1_train)
            l2_train_predict.append(accuracy_score(lr_l2.predict(x_train), g1_train))
            l2_test_predict.append(accuracy_score(lr_l2.predict(x_test), g1_test))

        data = [l1_train_predict, l2_train_predict, l1_test_predict, l2_test_predict]
        label = ['l1_train', 'l2_train', 'l1_test', "l2_test"]
        color = ['maroon', 'turquoise', 'dodgerblue', 'slategray']

        plt.rc('font', family='Times New Roman')
        plt.figure(figsize=(5, 4))
        for i in range(4):
            plt.plot(np.linspace(0.01, 2, 50), data[i], label=label[i], color=color[i], linewidth=2.5)

        plt.title('Accuracy with different C value for G1', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.xlabel('C value', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.ylabel('Accuracy', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.legend(loc="best")
        # plt.subplot(1,3,1)
        plt.show()

        ################## 测试不同c值下模型的表现情况: G2 ##################
        # 训练集表现
        l1_train_predict = []
        l2_train_predict = []
        # 测试集表现
        l1_test_predict = []
        l2_test_predict = []

        for c in np.linspace(0.01, 2, 50):
            lr_l1 = LogisticRegression(multi_class="ovr",penalty="l1", C=c, solver="liblinear", max_iter=1000)
            lr_l2 = LogisticRegression(multi_class="multinomial",penalty='l2', C=c, solver='newton-cg', max_iter=1000)

            # 训练模型，记录L1正则化模型在训练集测试集上的表现
            lr_l1.fit(x_train, g2_train)
            l1_train_predict.append(accuracy_score(lr_l1.predict(x_train), g2_train))
            l1_test_predict.append(accuracy_score(lr_l1.predict(x_test), g2_test))

            # 记录L2正则化模型的表现
            lr_l2.fit(x_train, g2_train)
            l2_train_predict.append(accuracy_score(lr_l2.predict(x_train), g2_train))
            l2_test_predict.append(accuracy_score(lr_l2.predict(x_test), g2_test))

        data = [l1_train_predict, l2_train_predict, l1_test_predict, l2_test_predict]
        label = ['l1_train', 'l2_train', 'l1_test', "l2_test"]
        color = ['maroon', 'turquoise', 'dodgerblue', 'slategray']

        plt.rc('font', family='Times New Roman')
        plt.figure(figsize=(5, 4))
        for i in range(4):
            plt.plot(np.linspace(0.01, 2, 50), data[i], label=label[i], color=color[i], linewidth=2.5)

        plt.title('Accuracy with different C value for G2', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.xlabel('C value', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.ylabel('Accuracy', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.legend(loc="best")
        # plt.subplot(1, 3, 2)
        plt.show()

        ################## 测试不同c值下模型的表现情况: G3 ##################
        # 训练集表现
        l1_train_predict = []
        l2_train_predict = []
        # 测试集表现
        l1_test_predict = []
        l2_test_predict = []

        for c in np.linspace(0.01, 2, 50):
            lr_l1 = LogisticRegression(multi_class="ovr",penalty="l1", C=c, solver="liblinear", max_iter=1000)
            lr_l2 = LogisticRegression(multi_class="multinomial",penalty='l2', C=c, solver='newton-cg', max_iter=1000)

            # 训练模型，记录L1正则化模型在训练集测试集上的表现
            lr_l1.fit(x_train, g3_train)
            l1_train_predict.append(accuracy_score(lr_l1.predict(x_train), g3_train))
            l1_test_predict.append(accuracy_score(lr_l1.predict(x_test), g3_test))

            # 记录L2正则化模型的表现
            lr_l2.fit(x_train, g3_train)
            l2_train_predict.append(accuracy_score(lr_l2.predict(x_train), g3_train))
            l2_test_predict.append(accuracy_score(lr_l2.predict(x_test), g3_test))

        data = [l1_train_predict, l2_train_predict, l1_test_predict, l2_test_predict]
        label = ['l1_train', 'l2_train', 'l1_test', "l2_test"]
        color = ['maroon', 'turquoise', 'dodgerblue', 'slategray']

        plt.rc('font', family='Times New Roman')
        plt.figure(figsize=(5, 4))
        for i in range(4):
            plt.plot(np.linspace(0.01, 2, 50), data[i], label=label[i], color=color[i], linewidth=2.5)

        plt.title('Accuracy with different C value for G3', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.xlabel('C value', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.ylabel('Accuracy', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.legend(loc="best")
        # plt.subplot(1, 3, 3)
        plt.show()
    else:
        print("Wrong Feature Length!!!")
        return 0
####### 32--decision tree--接口1个数据集 #######
def dt_draft(path,feature_length):
    print("######################### Decision Tree #########################")
    # 获得特征标签数据
    if (feature_length == 32):
        X, y = data_preprocessing_32(path)
        # 用最大似然估计自选新特征个数
        pca = PCA(n_components="mle")  # 实例化
        pca = pca.fit(X)  # 拟合模型
        X_dr = pca.transform(X)  # 获取新的特征
        print("Length of newly generated features: ")
        print(len(X_dr[0]))
        print("Explained variance: ")
        print(pca.explained_variance_)  # 查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
        print("Sum of explained variance ratio: ")
        print(pca.explained_variance_ratio_.sum())  # 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比之和，一般新特征能够反映原来特征的大部分信息（80%以上）即可
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_dr, y, test_size=0.1)
        clf = tree.DecisionTreeClassifier(max_depth=3)
        clf = clf.fit(X_train, y_train)
        clf.predict(X_test)
        print("<depth of decision tree> : ")
        print(clf.get_depth())  # 返回树的深度
        print("<number of leaves> : ")
        print(clf.get_n_leaves())  # 叶子节点个数
        print("<total node number> : ")
        print(clf.tree_.node_count)  # 总节点个数

        ####### 生成决策树文件 ######
        dot_data = tree.export_graphviz(clf, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("Decision Tree")
        # ratio = accuracy_score(clf.predict(X_test), y_test)
        ratio = clf.score(X_test, y_test)
        print("<Accuracy> : ")
        print(ratio)
    elif (feature_length == 30):
        X, G1, G2, G3 = data_preprocessing_30(path)
        # 用最大似然估计自选新特征个数
        pca = PCA(n_components="mle")  # 实例化
        pca = pca.fit(X)  # 拟合模型
        X_dr = pca.transform(X)  # 获取新的特征
        print("Length of newly generated features: ")
        print(len(X_dr[0]))
        print("Explained variance: ")
        print(pca.explained_variance_)  # 查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
        print("Sum of explained variance ratio: ")
        print(pca.explained_variance_ratio_.sum())  # 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比之和，一般新特征能够反映原来特征的大部分信息（80%以上）即可

        # 划分训练集和测试集
        X_train, X_test, G1_train, G1_test = train_test_split(X_dr, G1, test_size=0.1)
        X_train, X_test, G2_train, G2_test = train_test_split(X_dr, G2, test_size=0.1)
        X_train, X_test, G3_train, G3_test = train_test_split(X_dr, G3, test_size=0.1)

        print("## G1 model training and testing begin! ##")
        clf_g1 = tree.DecisionTreeClassifier(max_depth=3)
        clf_g1 = clf_g1.fit(X_train, G1_train)
        clf_g1.predict(X_test)
        print("<depth of G1 decision tree> : ")
        print(clf_g1.get_depth())  # 返回树的深度
        print("<number of G1 leaves> : ")
        print(clf_g1.get_n_leaves())  # 叶子节点个数
        print("<G1 total node number> : ")
        print(clf_g1.tree_.node_count)  # 总节点个数
        ####### 生成决策树文件 ######
        dot_data = tree.export_graphviz(clf_g1, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("Decision Tree for G1")
        # ratio = accuracy_score(clf.predict(X_test), y_test)
        ratio = clf_g1.score(X_test, G1_test)
        print("<Accuracy_G1> : ")
        print(ratio)
        print("## G1 done! ##")

        print("## G2 model training and testing begin! ##")
        clf_g2 = tree.DecisionTreeClassifier(max_depth=3)
        clf_g2 = clf_g2.fit(X_train, G2_train)
        clf_g2.predict(X_test)
        print("<depth of G2 decision tree> : ")
        print(clf_g2.get_depth())  # 返回树的深度
        print("<number of G2 leaves> : ")
        print(clf_g2.get_n_leaves())  # 叶子节点个数
        print("<G2 total node number>: ")
        print(clf_g2.tree_.node_count)  # 总节点个数
        ####### 生成决策树文件 ######
        dot_data = tree.export_graphviz(clf_g2, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("Decision Tree for G2")
        # ratio = accuracy_score(clf.predict(X_test), y_test)
        ratio = clf_g1.score(X_test, G2_test)
        print("<Accuracy_G2> : ")
        print(ratio)
        print("## G2 done! ##")

        print("## G3 model training and testing begin! ##")
        clf_g3 = tree.DecisionTreeClassifier(max_depth=3)
        clf_g3 = clf_g3.fit(X_train, G3_train)
        clf_g3.predict(X_test)
        print("<depth of G3 decision tree> : ")
        print(clf_g3.get_depth())  # 返回树的深度
        print("<number of G3 leaves> : ")
        print(clf_g3.get_n_leaves())  # 叶子节点个数
        print("<G3 total node number> : ")
        print(clf_g3.tree_.node_count)  # 总节点个数
        ####### 生成决策树文件 ######
        dot_data = tree.export_graphviz(clf_g3, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("Decision Tree for G3")
        # ratio = accuracy_score(clf.predict(X_test), y_test)
        ratio = clf_g1.score(X_test, G3_test)
        print("<Accuracy_G3> : ")
        print(ratio)
        print("## G3 done! ##")
    else:
        print("Wrong Feature Length!!!")
        return 0
