# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 11:26
# @Author  : Li Qi
# @FileName: DBSCAN&KMEANS_final.py
# @Function: final version of project3 to implement DBSCAN and KMEANS based on MNIST.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import mixture
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.cluster import adjusted_mutual_info_score
from time import time
import warnings
np.random.seed(23003)
random_state = 2018
warnings.filterwarnings('ignore')

def top1(lst):
    return max(lst, default='列表为空', key=lambda v: lst.count(v))

def MyAccuracyAndGeneralResults(clusterDF, labelsDF):
    countByCluster = pd.DataFrame(data=clusterDF['cluster'].value_counts())
    countByCluster.reset_index(inplace=True, drop=False)
    countByCluster.columns = ['cluster', 'clusterCount']
    preds = pd.concat([labelsDF, clusterDF], axis=1)
    preds.columns = ['trueLabel', 'cluster']
    countByLabel = pd.DataFrame(data=preds.groupby('trueLabel').count())
    countMostFreq = pd.DataFrame(data=preds.groupby('cluster').agg(lambda x: x.value_counts().iloc[0]))
    countMostFreq.reset_index(inplace=True, drop=False)
    countMostFreq.columns = ['cluster', 'countMostFrequent']
    accuracyDF = countMostFreq.merge(countByCluster,
                                     left_on="cluster", right_on="cluster")
    overallAccuracy = accuracyDF.countMostFrequent.sum() / accuracyDF.clusterCount.sum()
    accuracyByLabel = accuracyDF.countMostFrequent / accuracyDF.clusterCount
    return countByCluster, countByLabel, countMostFreq, accuracyDF, overallAccuracy, accuracyByLabel

def GenerateMeasures(n_clusters_, n_noise_, labels_true, labels):
    print(f'Estimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points: {n_noise_}')
    print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.4f}" )
    print(f"Completeness: {metrics.completeness_score(labels_true, labels):.4f}")
    print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.4f}")
    print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.4f}")
    print(f"Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels_true, labels):.4f}")

def ViewDigit(example_i):
    label = y.loc[example_i]
    image = X.loc[example_i,:].values.reshape([28,28])
    image = np.fliplr(image)  # 左右翻转
    img = np.rot90(image, 1)
    plt.title(f'Example: {example_i}  Label: {label}')
    plt.imshow(image)
    plt.show()

def ViewDigit_W(example_i, predict_label,X,y):
    label = y.loc[example_i]
    image = X.loc[example_i,:].values.reshape([28,28])
    image = np.fliplr(image)
    image = np.rot90(image, 1)
    plt.title(f'true_Label: {int(label)}   predict_Label: {predict_label}')
    plt.imshow(image)
    plt.show()

def GenerateMeasures_2(labels_true, labels):
    print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.4f}" )
    print(f"Completeness: {metrics.completeness_score(labels_true, labels):.4f}")
    print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.4f}")
    print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.4f}")
    print(f"Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels_true, labels):.4f}")

############################# 数据处理阶段 #############################
print("########################################## 数据读取阶段 ##########################################")
X = []
y = []
for i in range(10):
    a = np.loadtxt('num_%d.txt'%i)
    X = np.append(X,a)
X = np.reshape(X,(1000,784))
y = np.loadtxt('labels.txt')
X = np.float64(X)
y = np.float64(y)
random_state = check_random_state(0)  # 随机数种子
permutation = random_state.permutation(X.shape[0]) # 随机排列数组，打乱样本
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0],-1))
X = pd.DataFrame(data=X)
y = pd.Series(data=y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1) # 拆分验证集

scaler = StandardScaler() # 数据归一化
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)
print(f'Dimensions of Training Set: {X_train.shape}')
print(f'Dimensions of Labels for the Training Set: {y_train.shape}')
print(f'Dimensions of Testing Set: {X_test.shape}')
print(f'Dimensions of Labels for the Testing Set: {y_test.shape}')

print("########################################## 显示错误类别 ##########################################")
XX = []
yy = []
for i in range(10):
    a = np.loadtxt('num_%d.txt'%i)
    XX = np.append(XX,a)
XX = np.reshape(XX,(1000,784))
yy = np.loadtxt('labels.txt')
XX = np.float64(XX)
yy = np.float64(yy)

XX = XX.reshape((XX.shape[0],-1))

XX = pd.DataFrame(data=XX)
yy = pd.Series(data=yy)

pca = PCA(n_components= 784, random_state=23003)
XX_PCA = pca.fit_transform(XX)
XX_PCA = pd.DataFrame(data = XX_PCA)

n_components = 2
learning_rate = 300
perplexity = 30
early_exaggeration = 12
init = 'random'

tSNE = TSNE(n_components=n_components, learning_rate=learning_rate,
            perplexity=perplexity, early_exaggeration=early_exaggeration,
            init=init, random_state=random_state)
XX_train_tSNE = tSNE.fit_transform(XX_PCA.loc[:,:99])
XX_train_tSNE = pd.DataFrame(data=XX_train_tSNE)

kmeans = KMeans(n_clusters=10, n_init=10,
            max_iter=300, tol=0.0001, random_state=random_state,
            n_jobs=2)
kmeans = kmeans.fit(XX_train_tSNE)
yy_pre = kmeans.predict(XX_train_tSNE)


print("yy_pre: ")
print(yy_pre)
maxlabel = []
mid = (yy_pre[0:100]).tolist()
maxlabel.append(top1(mid))

mid = (yy_pre[100:200]).tolist()
maxlabel.append(top1(mid))

mid = (yy_pre[200:300]).tolist()
maxlabel.append(top1(mid))

mid = (yy_pre[300:400]).tolist()
maxlabel.append(top1(mid))

mid = (yy_pre[400:500]).tolist()
maxlabel.append(top1(mid))

mid = (yy_pre[500:600]).tolist()
maxlabel.append(top1(mid))

mid = (yy_pre[600:700]).tolist()
maxlabel.append(top1(mid))

mid = (yy_pre[700:800]).tolist()
maxlabel.append(top1(mid))

mid = (yy_pre[800:900]).tolist()
maxlabel.append(top1(mid))

mid = (yy_pre[900:1000]).tolist()
maxlabel.append(1)

print("maxlabel: ")
print(maxlabel)

wrong0 = []
wrong0_label = []
mid = yy_pre[0:100]
for i in range(100):
    if(mid[i]!=maxlabel[0]):
        wrong0.append(i)
        wrong0_label.append(mid[i])
print("wrong0: ")
print(wrong0)
print("wrong0_label: ")
print(wrong0_label)
print("第一个错误示例显示中...")
ViewDigit_W(wrong0[1],maxlabel.index(wrong0_label[1]),XX,yy)

wrong1 = []
wrong1_label = []
mid = yy_pre[100:200]
for i in range(100):
    if(mid[i]!=maxlabel[1]):
        wrong1.append(i)
        wrong1_label.append(mid[i])
print("wrong1: ")
print(wrong1)
print("wrong1_label: ")
print(wrong1_label)
print("第二个错误示例显示中...")
ViewDigit_W(wrong1[0]+100,maxlabel.index(wrong1_label[0]),XX,yy)

wrong2 = []
wrong2_label = []
mid = yy_pre[200:300]
for i in range(100):
    if(mid[i]!=maxlabel[2]):
        wrong2.append(i)
        wrong2_label.append(mid[i])
print("wrong2: ")
print(wrong2)
print("wrong2_label: ")
print(wrong2_label)
print("第三个错误示例显示中...")
ViewDigit_W(wrong2[0]+200,maxlabel.index(wrong2_label[0]),XX,yy)

wrong3 = []
wrong3_label = []
mid = yy_pre[300:400]
for i in range(100):
    if(mid[i]!=maxlabel[3]):
        wrong3.append(i)
        wrong3_label.append(mid[i])
print("wrong3: ")
print(wrong3)
print("wrong3_label: ")
print(wrong3_label)
print("第四个错误示例显示中...")
ViewDigit_W(wrong3[0]+300,maxlabel.index(wrong3_label[0]),XX,yy)

wrong4 = []
wrong4_label = []
mid = yy_pre[400:500]
for i in range(100):
    if(mid[i]!=maxlabel[4]):
        wrong4.append(i)
        wrong4_label.append(mid[i])
print("wrong4: ")
print(wrong4)
print("wrong4_label: ")
print(wrong4_label)
print("第五个错误示例显示中...")
ViewDigit_W(wrong4[0]+400,maxlabel.index(wrong4_label[0]),XX,yy)

wrong5 = []
wrong5_label = []
mid = yy_pre[500:600]
for i in range(100):
    if(mid[i]!=maxlabel[5]):
        wrong5.append(i)
        wrong5_label.append(mid[i])
print("wrong5: ")
print(wrong5)
print("wrong5_label: ")
print(wrong5_label)
print("第六个错误示例显示中...")
ViewDigit_W(wrong5[0]+500,maxlabel.index(wrong5_label[0]),XX,yy)

wrong6 = []
wrong6_label = []
mid = yy_pre[600:700]
for i in range(100):
    if(mid[i]!=maxlabel[6]):
        wrong6.append(i)
        wrong6_label.append(mid[i])
print("wrong6: ")
print(wrong6)
print("wrong6_label: ")
print(wrong6_label)
print("第七个错误示例显示中...")
ViewDigit_W(wrong6[0]+600,maxlabel.index(wrong6_label[0]),XX,yy)

wrong7 = []
wrong7_label = []
mid = yy_pre[700:800]
for i in range(100):
    if(mid[i]!=maxlabel[7]):
        wrong7.append(i)
        wrong7_label.append(mid[i])
print("wrong7: ")
print(wrong7)
print("wrong7_label: ")
print(wrong7_label)
print("第八个错误示例显示中...")
ViewDigit_W(wrong7[0]+700,maxlabel.index(wrong7_label[0]),XX,yy)


wrong8 = []
wrong8_label = []
mid = yy_pre[800:900]
for i in range(100):
    if(mid[i]!=maxlabel[8]):
        wrong8.append(i)
        wrong8_label.append(mid[i])
print("wrong8: ")
print(wrong8)
print("wrong8_label: ")
print(wrong8_label)
print("第九个错误示例显示中...")
ViewDigit_W(wrong8[0]+800,maxlabel.index(wrong8_label[0]),XX,yy)


wrong9 = []
wrong9_label = []
mid = yy_pre[900:1000]
for i in range(100):
    if(mid[i]!=maxlabel[9]):
        wrong9.append(i)
        wrong9_label.append(mid[i])
print("wrong9: ")
print(wrong9)
print("wrong9_label: ")
print(wrong9_label)
print("第十个错误示例显示中...")
ViewDigit_W(wrong9[0]+900,maxlabel.index(wrong9_label[0]),XX,yy)
# print("########################################## 显示错误类别结束! ##########################################")

# 手写数字数据实例
# print([ViewDigit(i) for i in [99, 199, 299,399,499,599,699,799,899]])

############################# 降维手段测试 #############################
print("########################################## 降维手段测试 ##########################################")

############降维手段：主成分分析(PCA)############
print("#################  降维手段：主成分分析(PCA)  #################")
pca = PCA(n_components= 784, random_state=23003)
X_PCA = pca.fit_transform(X)
X_PCA = pd.DataFrame(data = X_PCA)
print(f"Variance explained by all 784 principal components: {sum(pca.explained_variance_ratio_)}")
importanceOfPrincipalComponents = pd.DataFrame(data = pca.explained_variance_ratio_)
importanceOfPrincipalComponents = importanceOfPrincipalComponents.T
print(f"Variance explained by first 10 principal components: {importanceOfPrincipalComponents.loc[:,0:9].sum(axis=1).values}")
print(f"Variance explained by first 20 principal components: {importanceOfPrincipalComponents.loc[:,0:19].sum(axis=1).values}")
print(f"Variance explained by first 100 principal components: {importanceOfPrincipalComponents.loc[:,0:99].sum(axis=1).values}")
print(f"Variance explained by first 150 principal components: {importanceOfPrincipalComponents.loc[:,0:149].sum(axis=1).values}")
print(f"Variance explained by first 200 principal components: {importanceOfPrincipalComponents.loc[:,0:199].sum(axis=1).values}")
print(f"Variance explained by first 250 principal components: {importanceOfPrincipalComponents.loc[:,0:249].sum(axis=1).values}")
tempDF = pd.DataFrame(data=X_PCA.loc[:,0:1], index=X_PCA.index)
tempDF = pd.concat((tempDF,y.astype(int)), axis=1, join="inner")
tempDF.columns = ["First Loading", "Second Loading", "Label"]
sns.lmplot(x="First Loading", y="Second Loading", hue="Label",
           data=tempDF, fit_reg=False, aspect=1.2,height=4)
plt.grid()
print("图像显示中...")
plt.show()
print("图像已关闭")

############降维手段：奇异值分解(SVD)############
print("#################  降维手段：奇异值分解(SVD)  #################")
n_components = 200
algorithm = 'randomized'
n_iter = 5

svd = TruncatedSVD(n_components=n_components, algorithm=algorithm,
                   n_iter=n_iter, random_state=random_state)

X_SVD = svd.fit_transform(X)
X_SVD = pd.DataFrame(data=X_SVD)
tempDF = pd.DataFrame(data=X_SVD.loc[:,0:1], index=X_SVD.index)
tempDF = pd.concat((tempDF,y.astype(int)), axis=1, join="inner")
tempDF.columns = ["First Loading", "Second Loading", "Label"]
sns.lmplot(x="First Loading", y="Second Loading", hue="Label",
           data=tempDF, fit_reg=False, aspect=1.2,height=4)
plt.grid()
print("图像显示中...")
plt.show()
print("图像已关闭")

############降维手段：t分布随机邻域嵌入(t-SNE)############
print("#################  降维手段：t分布随机邻域嵌入(t-SNE)  #################")
n_components = 2
learning_rate = 300
perplexity = 30
early_exaggeration = 12
init = 'random'

tSNE = TSNE(n_components=n_components, learning_rate=learning_rate,
            perplexity=perplexity, early_exaggeration=early_exaggeration,
            init=init, random_state=random_state)

X_train_tSNE = tSNE.fit_transform(X_PCA.loc[:,:99])
X_train_tSNE = pd.DataFrame(data=X_train_tSNE)

tempDF = pd.DataFrame(data=X_train_tSNE.loc[:,0:1], index=X_PCA.index)
tempDF = pd.concat((tempDF,y.astype(int)), axis=1, join="inner")
tempDF.columns = ["First Loading", "Second Loading", "Label"]

sns.lmplot(x="First Loading", y="Second Loading", hue="Label",
           data=tempDF, fit_reg=False, aspect=1.2,height=4)
plt.grid()
print("图像显示中...")
plt.show()
print("图像已关闭")

################################ Metrics for clustering methods ################################
print("################################ 聚类方法的评价指标 ################################")
print("################## Silhouette-Score & AMI for K-MEANS ##################")
inertia_scores = []
sil_scores = []
for n in range(2, 13):
    km = KMeans(n_clusters=n).fit(X_train_tSNE)
    inertia_scores.append(km.inertia_)
    # 轮廓系数接收的参数中，第二个参数至少有两个分类
    sc = silhouette_score(X_train_tSNE, km.labels_)
    ami = adjusted_mutual_info_score(y,km.labels_)
    sil_scores.append(sc)
    print("n_clusters: {}\tsilhouette_score: {}\tAMI: {}".format(
        n, sc, ami))

print("################## Silhouette-Score & AMI for DBSCAN ##################")
for n in range(3, 8):
    min_samples = 15
    db = DBSCAN(eps=n, min_samples=15, metric='l2')
    db = db.fit(X_train_tSNE)

    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise_ = list(db.labels_).count(-1)
    X_train_tSNE_dbscanClustered = db.fit_predict(X_train_tSNE)
    X_train_tSNE_dbscanClustered = pd.DataFrame(data=X_train_tSNE_dbscanClustered, columns=['cluster'])
    ami = adjusted_mutual_info_score(y,db.labels_)
    sc = silhouette_score(X_train_tSNE, db.labels_)
    sil_scores.append(sc)
    print("n_eps: {}\tsilhouette_score: {}\tAMI: {}".format(
        n, sc, ami))

print("########################################## 模型测试 ##########################################")
print("#################  模型设置：DBSCAN on X_PCA  #################")
eps = 5 # try different values, but with no improvement...
min_samples = 5
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_PCA.loc[:,0:199]) # try different values of loadings

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise_ = list(db.labels_).count(-1)
GenerateMeasures(n_clusters_, n_noise_, y, db.labels_)

db = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_PCA.loc[:,0:199])
XdbscanClustered = pd.DataFrame(data=db, index = X_PCA.index, columns=['cluster'])
countByCluster, countByLabel, countMostFreq, accuracyDF, overallAccuracy, accuracyByLabel = MyAccuracyAndGeneralResults(XdbscanClustered, y)
print(f"The overall Accuracy with dbscan on X_PCA is: {overallAccuracy}")
print(f"The clusters found are:")
print(countByCluster)
# print("#################  DBSCAN on X_PCA: Done!  #################")

print("#################  模型设置：DBSCAN on X_svd  #################")
eps = 50
min_samples = 50
# try different values
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_SVD)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise_ = list(db.labels_).count(-1)

GenerateMeasures(n_clusters_, n_noise_, y, db.labels_)

db = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_SVD)
XdbscanClustered = pd.DataFrame(data=db, index = X_PCA.index, columns=['cluster'])
countByCluster, countByLabel, countMostFreq, accuracyDF, overallAccuracy, accuracyByLabel = MyAccuracyAndGeneralResults(XdbscanClustered, y)
print(f"The overall Accuracy with dbscan on X_svd is: {overallAccuracy}")
print(f"The clusters found are:")
print(countByCluster)
# print("#################  DBSCAN on X_svd: Done!  #################")

print("#################  模型设置：DBSCAN on X_tSNE  #################")
eps = 5
min_samples = 15

db = DBSCAN(eps=eps, min_samples=min_samples, metric = 'l2')
# distance tested: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
db = db.fit(X_train_tSNE)

n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise_ = list(db.labels_).count(-1)

GenerateMeasures(n_clusters_, n_noise_, y, db.labels_)

X_train_tSNE_dbscanClustered = db.fit_predict(X_train_tSNE)
X_train_tSNE_dbscanClustered = pd.DataFrame(data=X_train_tSNE_dbscanClustered,columns=['cluster'])

countByCluster_dbscan_tSNE, countByLabel_dbscan_tSNE, countMostFreq_dbscan_tSNE, accuracyDF_dbscan_tSNE, overallAccuracy_dbscan_tSNE, accuracyByLabel_dbscan_tSNE =  MyAccuracyAndGeneralResults(X_train_tSNE_dbscanClustered, y)

print(f"The overall Accuracy with dbscan on X_tSNE is: {overallAccuracy_dbscan_tSNE}")
print(f"The clusters found are:")
print(countByCluster_dbscan_tSNE)
# print("#################  DBSCAN on X_tSNE: Done!  #################")

print("#################  模型设置：K-MEANS on X_PCA  #################")
n_init = 10
max_iter = 300
random_state = 20023
tol = 0.0001
n_jobs = 2

overallAccuracy_kMeansDF =  pd.DataFrame(data=[], index=range(2,21),columns=['overallAccuracy'])
for n_clusters in range(2,21):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init,
                max_iter=max_iter, tol=tol, random_state=random_state,
                n_jobs=n_jobs)

    kmeans.fit(X_PCA.loc[:,:199])
    X_train_kmeansClustered = kmeans.predict(X_PCA.loc[:,0:199])
    X_train_kmeansClustered =  pd.DataFrame(data=X_train_kmeansClustered, index=X_PCA.index,
                     columns=['cluster'])

    countByCluster_kMeans, countByLabel_kMeans, countMostFreq_kMeans, accuracyDF_kMeans, overallAccuracy_kMeans, accuracyByLabel_kMeans = MyAccuracyAndGeneralResults(X_train_kmeansClustered, y)

    overallAccuracy_kMeansDF.loc[n_clusters] = overallAccuracy_kMeans
print(overallAccuracy_kMeansDF)

plt.figure(figsize=(5,5))
plt.title("Overall Accuracy in K Means on PCA reduced dataset")
plt.xlabel("n_clusters")
sns.lineplot(x = overallAccuracy_kMeansDF.index, y = overallAccuracy_kMeansDF.overallAccuracy,
            estimator=None, lw=1, label="Accuracy")
plt.grid()
print("图像显示中...")
plt.show()
print("图像已关闭")

# print("#################  K-MEANS on X_PCA: Done!  #################")

print("#################  模型设置：K-MEANS on X_tSNE  #################")
kmeans = KMeans(n_clusters=10, n_init=n_init,
                max_iter=max_iter, tol=tol, random_state=random_state,
                n_jobs=n_jobs)
kmeans.fit(X_PCA.loc[:,:199])
X_train_kmeansClustered = kmeans.predict(X_PCA.loc[:,0:199])
X_train_kmeansClustered =  pd.DataFrame(data=X_train_kmeansClustered, index=X_PCA.index,
                 columns=['cluster'])
labels = kmeans.labels_
labels_true = y
GenerateMeasures_2(labels_true, labels)

overallAccuracy_kMeansDF =  pd.DataFrame(data=[],index=range(2,21),columns=['overallAccuracy'])

for n_clusters in range(2,21):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init,
                max_iter=max_iter, tol=tol, random_state=random_state,
                n_jobs=n_jobs)

    kmeans.fit(X_train_tSNE)
    X_train_kmeansClustered = kmeans.predict(X_train_tSNE)
    X_train_kmeansClustered =  pd.DataFrame(data=X_train_kmeansClustered, index=X_PCA.index,
                     columns=['cluster'])

    countByCluster_kMeans, countByLabel_kMeans, countMostFreq_kMeans, accuracyDF_kMeans, overallAccuracy_kMeans, accuracyByLabel_kMeans = MyAccuracyAndGeneralResults(X_train_kmeansClustered, y)

    overallAccuracy_kMeansDF.loc[n_clusters] = overallAccuracy_kMeans
print(overallAccuracy_kMeansDF)

plt.figure(figsize=(5,5))
plt.title("Overall Accuracy in K Mean on t-SNE reduced dataset")
plt.xlabel("n_clusters")
sns.lineplot(x = overallAccuracy_kMeansDF.index, y = overallAccuracy_kMeansDF.overallAccuracy,
            estimator=None, lw=1, label="Accuracy")
plt.grid()
print("图像显示中...")
plt.show()
print("图像已关闭")

kmeans = KMeans(n_clusters=10, n_init=n_init,
            max_iter=max_iter, tol=tol, random_state=random_state,
            n_jobs=n_jobs)

kmeans.fit(X_train_tSNE)
X_train_kmeansClustered = kmeans.predict(X_train_tSNE)

labels = kmeans.labels_
labels_true = y
GenerateMeasures_2(labels_true, labels)
# print("#################  K-MEANS on X_tSNE: Done!  #################")

print("#################  模型设置：GM on X_PCA  #################")
clf = mixture.GaussianMixture(n_components=10, covariance_type='full')
X_train_GM_Clustered = clf.fit(X_PCA.loc[:,:1])
X_train_GM_Clustered = clf.predict(X_PCA.loc[:,:1])
X_train_GM_Clustered =  pd.DataFrame(data=X_train_GM_Clustered, index=X_train_tSNE.index,
                     columns=['cluster'])

countByCluster_GM, countByLabel_GM, countMostFreq_GM, accuracyDF_GM, overallAccuracy_GM, accuracyByLabel_GM = MyAccuracyAndGeneralResults(X_train_GM_Clustered, y)
print(overallAccuracy_GM)

GenerateMeasures_2(labels_true, X_train_GM_Clustered.cluster)
print(countByCluster_GM)
# print("#################  GM on X_PCA: Done!  #################")

print("####################################### 聚类结果示例 #######################################")
tempDF["KmeansLabel"] = X_train_kmeansClustered
sns.lmplot(x="First Loading", y="Second Loading", hue="KmeansLabel",
           data=tempDF, fit_reg=False,aspect=1.2,height=5.5)
plt.gcf().subplots_adjust(top = 0.953)
plt.title("K-means clusters")
plt.grid()
print("图像显示中...")
plt.show()
print("图像已关闭")

tempDF["GMlabel"] = X_train_GM_Clustered.cluster
sns.lmplot(x="First Loading", y="Second Loading", hue="GMlabel",
           data=tempDF, fit_reg=False, aspect=1.2,height=5.5)
plt.gcf().subplots_adjust(top = 0.953)
plt.title("Gaussian Mixture clusters")
plt.grid()
print("图像显示中...")
plt.show()
print("图像已关闭")

tempDF["DBSCANlabel"] = X_train_tSNE_dbscanClustered.cluster
sns.lmplot(x="First Loading", y="Second Loading", hue="DBSCANlabel",
           data=tempDF, fit_reg=False, aspect=1.2,height=5.5)
plt.gcf().subplots_adjust(top = 0.953)
plt.title("DBSCAN clusters")
plt.grid()
print("图像显示中...")
plt.show()
print("图像已关闭")











