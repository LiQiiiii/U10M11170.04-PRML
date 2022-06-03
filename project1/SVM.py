from sklearn import svm
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV #网格搜索
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

with open ("prml.txt") as file:
    data = file.readlines()
    number_line = len(data)
for i in range(number_line):
    data[i] = data[i].strip()
    data[i] = data[i].split(" ")
X = np.array(data) # 提取data并转化为nparray
X = StandardScaler().fit_transform(X)
print('X: ')
print(X)

with open ("prml_label.txt") as file:
    data = file.readlines()
    number_line = len(data)
for i in range(number_line):
    data[i] = data[i].strip()
    data[i] = data[i].split(" ")
y = np.array(data).ravel() # 提取data并转化为nparray
print('y: ')
print(y)

seed = 7 #重现随机生成的训练
test_size = 0.33 #33%测试，67%训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


clf = svm.SVC(C=10,kernel='rbf',gamma=0.01)
clf.fit(X_train, y_train)
# 网格搜索寻找最佳参数
grid = GridSearchCV(clf, param_grid={'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}, cv=4)
# 找到的最佳参数为C=10,gamma=0.01
grid.fit(X, y)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


print('----support vector indice----')
print(clf.support_)
print('------ support vector ------')
print(clf.support_vectors_)
# print('------- coef of feacture -------')
# print(clf.coef_)
print('--------- intercept_ -------')
print(clf.intercept_)
print('--------- n_support_ -------')
print(clf.n_support_)

print('--------- predict_result_ -------')
print('0:male, 1:female')
pre = []
for i in range(len(X_test)):
    predict = clf.predict([X_test[i]])
    pre.append(predict[0])

print('predict result: ')
print(pre)

score = 0

for i in range(len(pre)):
    if pre[i] == y_test[i]:
        score += 1
rate = score / len(pre)
print('predict accuracy rate: ')
print(rate)








