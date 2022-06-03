# -*- coding: utf-8 -*-
# @Time    : 2022/6/2 19:58
# @Author  : Li Qi
# @FileName: model_TwoPath.py
# @Function: three models with two input paths

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

####### 32-knn-final #######
def knn(trainset_path,testset_path,feature_length):
    print("######################### KNN #########################")
    # 获得特征标签数据
    if(feature_length == 32):
        X_train, y_train = data_preprocessing_32(trainset_path)
        X_test, y_test = data_preprocessing_32(testset_path)
        # 用最大似然估计自选新特征个数
        pca = PCA(n_components="mle")  # 实例化
        pca = pca.fit(X_train)  # 拟合模型
        X_train = pca.transform(X_train)  # 获取新的特征
        X_test = pca.transform(X_test)
        print("Length of newly generated features: ")
        print(len(X_train[0]))
        print("Explained variance: ")
        print(pca.explained_variance_)  # 查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
        print("Sum of explained variance ratio: ")
        print(pca.explained_variance_ratio_.sum())  # 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比之和，一般新特征能够反映原来特征的大部分信息（80%以上）即可

        ##### 拟合模型 #####
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X_train, y_train)

        Accuracy = accuracy_score(knn.predict(X_test), y_test)
        print('Accuracy: ')
        print(Accuracy)
    elif(feature_length == 30):
        X_train, g1_train, g2_train, g3_train = data_preprocessing_30(trainset_path)
        X_test, g1_test, g2_test, g3_test = data_preprocessing_30(testset_path)
        # 用最大似然估计自选新特征个数
        pca = PCA(n_components="mle")  # 实例化
        pca = pca.fit(X_train)  # 拟合模型
        X_train = pca.transform(X_train)  # 获取新的特征
        X_test = pca.transform(X_test)
        print("Length of newly generated features: ")
        print(len(X_train[0]))
        print("Explained variance: ")
        print(pca.explained_variance_)  # 查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
        print("Sum of explained variance ratio: ")
        print(pca.explained_variance_ratio_.sum())  # 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比之和，一般新特征能够反映原来特征的大部分信息（80%以上）即可

        print("G1 model training and testing begin!")
        ##### 拟合模型 #####
        knn_g1 = KNeighborsClassifier(n_neighbors=7)
        knn_g1.fit(X_train, g1_train)
        Accuracy_g1 = accuracy_score(knn_g1.predict(X_test), g1_test)
        print('Accuracy_G1: ')
        print(Accuracy_g1)
        print("G1 done!")

        print("G2 model training and testing begin!")
        ##### 拟合模型 #####
        knn_g2 = KNeighborsClassifier(n_neighbors=7)
        knn_g2.fit(X_train, g2_train)
        Accuracy_g2 = accuracy_score(knn_g2.predict(X_test), g2_test)
        print('Accuracy_G2: ')
        print(Accuracy_g2)
        print("G2 done!")

        print("G3 model training and testing begin!")
        ##### 拟合模型 #####
        knn_g3 = KNeighborsClassifier(n_neighbors=7)
        knn_g3.fit(X_train, g3_train)
        Accuracy_g3 = accuracy_score(knn_g3.predict(X_test), g3_test)
        print('Accuracy_G3: ')
        print(Accuracy_g3)
        print("G3 done")
    else:
        print("Wrong Feature Length!!!")
        return 0

####### 32-lg-final #######
def logistic_regression(trainset_path,testset_path, feature_length):
    print("######################### Logistic Regression #########################")
    # 获得特征标签数据
    if(feature_length == 32):
        X_train, y_train = data_preprocessing_32(trainset_path)
        X_test, y_test = data_preprocessing_32(testset_path)
        lr_l1 = LogisticRegression(multi_class="ovr", penalty="l1", C=1, solver="liblinear")
        lr_l2 = LogisticRegression(multi_class="multinomial", penalty="l2", C=0.5, solver="newton-cg")

        # 训练模型
        lr_l1.fit(X_train, y_train)
        lr_l2.fit(X_train, y_train)

        Accuracy1 = accuracy_score(lr_l1.predict(X_test), y_test)
        Accuracy2 = accuracy_score(lr_l2.predict(X_test), y_test)
        print("Accuracy for lr.l1: ")
        print(Accuracy1)
        print("Accuracy for lr.l2: ")
        print(Accuracy2)

        ## 测试不同c值下模型的表现情况
        # 训练集表现
        l1_train_predict = []
        l2_train_predict = []
        # 测试集表现
        l1_test_predict = []
        l2_test_predict = []

        for c in np.linspace(0.01, 2, 50):
            lr_l1 = LogisticRegression(multi_class="ovr", penalty="l1", C=c, solver="liblinear")
            lr_l2 = LogisticRegression(multi_class="multinomial", penalty="l2", C=c, solver="newton-cg")

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
        X_train, G1_train, G2_train, G3_train = data_preprocessing_30(trainset_path)
        X_test, G1_test, G2_test, G3_test = data_preprocessing_30(testset_path)


        print("G1 model training and testing begin!")
        # 训练模型
        lr_l1_g1 = LogisticRegression(multi_class="ovr", penalty="l1", C=1, solver="liblinear")
        lr_l2_g1 = LogisticRegression(multi_class="multinomial", penalty="l2", C=0.5, solver="newton-cg")
        lr_l1_g1.fit(X_train, G1_train)
        lr_l2_g1.fit(X_train, G1_train)
        Accuracy1 = accuracy_score(lr_l1_g1.predict(X_test), G1_test)
        Accuracy2 = accuracy_score(lr_l2_g1.predict(X_test), G1_test)
        print("Accuracy_G1 for lr.l1: ")
        print(Accuracy1)
        print("Accuracy_G1 for lr.l2: ")
        print(Accuracy2)
        print("G1 done!")

        print("G2 model training and testing begin!")
        # 训练模型
        lr_l1_g2 = LogisticRegression(multi_class="ovr", penalty="l1", C=1, solver="liblinear")
        lr_l2_g2 = LogisticRegression(multi_class="multinomial", penalty="l2", C=0.5, solver="newton-cg")
        lr_l1_g2.fit(X_train, G2_train)
        lr_l2_g2.fit(X_train, G2_train)
        Accuracy1 = accuracy_score(lr_l1_g2.predict(X_test), G2_test)
        Accuracy2 = accuracy_score(lr_l2_g2.predict(X_test), G2_test)
        print("Accuracy_G2 for lr.l1: ")
        print(Accuracy1)
        print("Accuracy_G2 for lr.l2: ")
        print(Accuracy2)
        print("G2 done!")

        print("G3 model training and testing begin!")
        # 训练模型
        lr_l1_g3 = LogisticRegression(multi_class="ovr", penalty="l1", C=1, solver="liblinear")
        lr_l2_g3 = LogisticRegression(multi_class="multinomial", penalty="l2", C=0.5, solver="newton-cg")
        lr_l1_g3.fit(X_train, G3_train)
        lr_l2_g3.fit(X_train, G3_train)
        Accuracy1 = accuracy_score(lr_l1_g3.predict(X_test), G3_test)
        Accuracy2 = accuracy_score(lr_l2_g3.predict(X_test), G3_test)
        print("Accuracy_G3 for lr.l1: ")
        print(Accuracy1)
        print("Accuracy_G3 for lr.l2: ")
        print(Accuracy2)
        print("G3 done!")

        ## 测试不同c值下模型的表现情况
        # 训练集表现
        l1_train_predict = []
        l2_train_predict = []
        # 测试集表现
        l1_test_predict = []
        l2_test_predict = []

        for c in np.linspace(0.01, 2, 50):
            lr_l1 = LogisticRegression(multi_class="ovr", penalty="l1", C=c, solver="liblinear")
            lr_l2 = LogisticRegression(multi_class="multinomial", penalty="l2", C=c, solver="newton-cg")

            # 训练模型，记录L1正则化模型在训练集测试集上的表现
            lr_l1.fit(X_train, G1_train)
            l1_train_predict.append(accuracy_score(lr_l1.predict(X_train), G1_train))
            l1_test_predict.append(accuracy_score(lr_l1.predict(X_test), G1_test))

            # 记录L2正则化模型的表现
            lr_l2.fit(X_train, G1_train)
            l2_train_predict.append(accuracy_score(lr_l2.predict(X_train), G1_train))
            l2_test_predict.append(accuracy_score(lr_l2.predict(X_test), G1_test))

        data = [l1_train_predict, l2_train_predict, l1_test_predict, l2_test_predict]
        label = ['l1_train', 'l2_train', 'l1_test', "l2_test"]
        color = ['maroon', 'turquoise', 'dodgerblue', 'slategray']

        plt.rc('font', family='Times New Roman')
        plt.figure(figsize=(5, 4))
        for i in range(4):
            plt.plot(np.linspace(0.01, 2, 50), data[i], label=label[i], color=color[i], linewidth=2.5)

        plt.title('Accuracy_G1 with different C value', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.xlabel('C value', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.ylabel('Accuracy', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.legend(loc="best")
        plt.show()

        ## 测试不同c值下模型的表现情况
        # 训练集表现
        l1_train_predict = []
        l2_train_predict = []
        # 测试集表现
        l1_test_predict = []
        l2_test_predict = []

        for c in np.linspace(0.01, 2, 50):
            lr_l1 = LogisticRegression(multi_class="ovr", penalty="l1", C=c, solver="liblinear")
            lr_l2 = LogisticRegression(multi_class="multinomial", penalty="l2", C=c, solver="newton-cg")

            # 训练模型，记录L1正则化模型在训练集测试集上的表现
            lr_l1.fit(X_train, G2_train)
            l1_train_predict.append(accuracy_score(lr_l1.predict(X_train), G2_train))
            l1_test_predict.append(accuracy_score(lr_l1.predict(X_test), G2_test))

            # 记录L2正则化模型的表现
            lr_l2.fit(X_train, G2_train)
            l2_train_predict.append(accuracy_score(lr_l2.predict(X_train), G2_train))
            l2_test_predict.append(accuracy_score(lr_l2.predict(X_test), G2_test))

        data = [l1_train_predict, l2_train_predict, l1_test_predict, l2_test_predict]
        label = ['l1_train', 'l2_train', 'l1_test', "l2_test"]
        color = ['maroon', 'turquoise', 'dodgerblue', 'slategray']

        plt.rc('font', family='Times New Roman')
        plt.figure(figsize=(5, 4))
        for i in range(4):
            plt.plot(np.linspace(0.01, 2, 50), data[i], label=label[i], color=color[i], linewidth=2.5)

        plt.title('Accuracy_G2 with different C value', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.xlabel('C value', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.ylabel('Accuracy', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.legend(loc="best")
        plt.show()

        ## 测试不同c值下模型的表现情况
        # 训练集表现
        l1_train_predict = []
        l2_train_predict = []
        # 测试集表现
        l1_test_predict = []
        l2_test_predict = []

        for c in np.linspace(0.01, 2, 50):
            lr_l1 = LogisticRegression(multi_class="ovr", penalty="l1", C=c, solver="liblinear")
            lr_l2 = LogisticRegression(multi_class="multinomial", penalty="l2", C=c, solver="newton-cg")

            # 训练模型，记录L1正则化模型在训练集测试集上的表现
            lr_l1.fit(X_train, G3_train)
            l1_train_predict.append(accuracy_score(lr_l1.predict(X_train), G3_train))
            l1_test_predict.append(accuracy_score(lr_l1.predict(X_test), G3_test))

            # 记录L2正则化模型的表现
            lr_l2.fit(X_train, G3_train)
            l2_train_predict.append(accuracy_score(lr_l2.predict(X_train), G3_train))
            l2_test_predict.append(accuracy_score(lr_l2.predict(X_test), G3_test))

        data = [l1_train_predict, l2_train_predict, l1_test_predict, l2_test_predict]
        label = ['l1_train', 'l2_train', 'l1_test', "l2_test"]
        color = ['maroon', 'turquoise', 'dodgerblue', 'slategray']

        plt.rc('font', family='Times New Roman')
        plt.figure(figsize=(5, 4))
        for i in range(4):
            plt.plot(np.linspace(0.01, 2, 50), data[i], label=label[i], color=color[i], linewidth=2.5)

        plt.title('Accuracy_G3 with different C value', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.xlabel('C value', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.ylabel('Accuracy', fontdict={'family': 'Times New Roman', 'size': 13})
        plt.legend(loc="best")
        plt.show()
    else:
        print("Wrong Feature Length!!!")
        return 0

####### 32-dt-final #######
def decision_tree(trainset_path,testset_path, feature_length):
    print("######################### Decision Tree #########################")
    # 获得特征标签数据
    if (feature_length == 32):
        X_train, y_train = data_preprocessing_32(trainset_path)
        X_test, y_test = data_preprocessing_32(testset_path)

        # 用最大似然估计自选新特征个数
        pca = PCA(n_components="mle")  # 实例化
        pca = pca.fit(X_train)  # 拟合模型
        X_train = pca.transform(X_train)  # 获取新的特征
        X_test = pca.transform(X_test)

        print("<Length of newly generated features> : ")
        print(len(X_train[0]))
        print("<Explained variance> : ")
        print(pca.explained_variance_)  # 查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
        print("<Sum of explained variance ratio> : ")
        print(pca.explained_variance_ratio_.sum())  # 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比之和，一般新特征能够反映原来特征的大部分信息（80%以上）即可

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
        X_train, G1_train, G2_train, G3_train = data_preprocessing_30(trainset_path)
        X_test, G1_test, G2_test, G3_test = data_preprocessing_30(testset_path)

        # 用最大似然估计自选新特征个数
        pca = PCA(n_components="mle")  # 实例化
        pca = pca.fit(X_train)  # 拟合模型
        X_train = pca.transform(X_train)  # 获取新的特征
        X_test = pca.transform(X_test)

        print("<Length of newly generated features> : ")
        print(len(X_train[0]))
        print("<Explained variance> : ")
        print(pca.explained_variance_)  # 查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
        print("<Sum of explained variance ratio> : ")
        print(pca.explained_variance_ratio_.sum())  # 查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比之和，一般新特征能够反映原来特征的大部分信息（80%以上）即可

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