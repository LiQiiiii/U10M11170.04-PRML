# -*- coding: utf-8 -*-
# @Time    : 2022/6/5 18:03
# @Author  : Li Qi
# @FileName: KMEANS_bad_performance.py
# @Function: A draft version of implementation of KMEANS, which has low accuracy.

# 导入相关包
from time import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import silhouette_score, silhouette_samples
# silhouette_score 返回所有样本的轮廓系数的均值
# silhouette_samples 返回每个样本的轮廓系数


def bench_k_means(kmeans, name, data, labels):
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))

# 数据载入模块
X = []
y = []
for i in range(10):
    a = np.loadtxt('num_%d.txt'%i)
    X = np.append(X,a)
X = np.reshape(X,(1000,784))
# y = str(np.loadtxt('labels.txt'))
y = np.loadtxt('labels.txt')
print("<An example of images: >")
print(X[0])
print("<Labels: >")
print(y)
X = np.float64(X)
y = np.float64(y)

"""
在matplotlib中，整个图像为一个Figure对象。在Figure对象中可以包含一个或者多个Axes对象。
每个Axes(ax)对象都是一个拥有自己坐标系统的绘图区域。
"""
fig, ax = plt.subplots(nrows=2,ncols=5)
ax = ax.flatten()  # flatten()将ax由2*5的Axes组展平成1*10的Axes组
for i in range(10):
    img = X[i+100*i].reshape(28, 28) # 将img从1*784重构成28*28的数组
    img = np.fliplr(img)  # 左右翻转
    img = np.rot90(img, 1)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_xlabel("Class %i" % i)
plt.tight_layout() # 作用是自动调整子图参数,使之填充整个图像区域
plt.show()

random_state = check_random_state(0)  # 随机数种子
permutation = random_state.permutation(X.shape[0]) # 随机排列数组，打乱样本
X = X[permutation]
y = y[permutation]

X = X.reshape((X.shape[0],-1))

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1) # 拆分验证集
print(x_train.shape)

scaler = StandardScaler() # 数据归一化
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train[0])
print(x_test[0])


print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

kmeans = KMeans(init="k-means++", n_clusters=10, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", data=x_train, labels=y_train)

kmeans = KMeans(init="random", n_clusters=10,  n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="random", data=x_train, labels=y_train)

pca = PCA(n_components=10).fit(x_train)
kmeans = KMeans(init=pca.components_, n_clusters=10, n_init=1)
bench_k_means(kmeans=kmeans, name="PCA-based", data=x_train, labels=y_train)

print(82 * "_")

reduced_data = PCA(n_components=2).fit_transform(x_train)
kmeans = KMeans(init="k-means++", n_clusters=10, n_init=4)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on the MNIST dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

inertia_scores = []
sil_scores = []
for n in range(2, 11):
    km = KMeans(n_clusters=n).fit(X, y)

    inertia_scores.append(km.inertia_)

    # 轮廓系数接收的参数中，第二个参数至少有两个分类
    sc = silhouette_score(X, km.labels_)
    sil_scores.append(sc)
    print("n_clusters: {}\tinertia: {}\tsilhoutte_score: {}".format(
        n, km.inertia_, sc))

kmeans1 = KMeans(init="k-means++", n_clusters=10, n_init=4, random_state=0)
kmeans1.fit(x_train, y_train)
Accuracy = accuracy_score(kmeans1.predict(x_test), y_test)
print(kmeans1.predict(x_train))
print(y_train)
