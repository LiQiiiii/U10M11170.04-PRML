# -*- coding: utf-8 -*-
# @Time    : 2022/6/5 18:54
# @Author  : Li Qi
# @FileName: MNIST_imageFlatten.py
# @Function: Save bmp image as txt with 1×784 array from 0 to 9
#-*- coding: utf-8 -*-
import os
import numpy
from PIL import Image   #导入Image模块
from pylab import *     #导入savetxt模块

#此函数读取特定文件夹下的bmp格式图像
def get_imlist(path):

    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]

def img_to_txt():
    for n in range(10):
        c = get_imlist(r"C:\Users\61060\Desktop\NPU-LQ\大三下\模式识别与机器学习\工程3+2019302863+李奇\project3\MNIST\%d"%n)  # r""是防止字符串转译
        print(c)  # 这里以list形式输出bmp格式的所有图像（带路径）
        d = len(c)  # 这可以以输出图像个数
        data = numpy.empty((d, 28 * 28))  # 建立d*（28*28）的矩阵
        while d > 0:
            img = Image.open(c[d - 1])  # 打开图像
            img_ndarray = numpy.asarray(img, dtype='float64')  # 将图像转化为数组
            data[d - 1] = numpy.ndarray.flatten(img_ndarray)  # 将图像的矩阵形式转化为一维数组保存到data中
            d = d - 1
        print(data)

        with open('num_%d.txt'%n, 'ab') as f:
            for i in range(100):
                A = numpy.array(data[i]).reshape(1, 784)
                savetxt(f, A, fmt="%.0f")  # 将数据保存到txt文件中
    with open("labels.txt", "w") as f:
        for i in range(10):
            for j in range(100):
                f.write(('%d\n' % i))

img_to_txt()

