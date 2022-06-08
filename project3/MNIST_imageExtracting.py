# -*- coding: utf-8 -*-
# @Time    : 2022/6/5 16:17
# @Author  : Li Qi
# @FileName: MNIST_imageExtracting.py
# @Function: Extract images from mnist and save as bmp type.

import os
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

'''
    mnist_dir   mnist数据集存储的路径
    save_dir    提取结果存储的目录
'''
def extract_mnist(mnist_dir, save_dir):
    rows = 28
    cols = 28

    # 加载mnist数据集
    mnist = input_data.read_data_sets(mnist_dir)
    # 获取训练图片数量
    shape = mnist.train.images.shape
    images_train_count = shape[0]
    pixels_count_per_image = shape[1]
    # 获取训练标签数量=训练图片数量
    labels = mnist.train.labels
    labels_train_count = labels.shape[0]

    if (images_train_count == labels_train_count):
        print("训练集共包含%d张图片，%d个标签" % (images_train_count, labels_train_count))
        print("每张图片包含%d个像素" % (pixels_count_per_image))
        print("数据类型为", mnist.train.images.dtype)

        # mnist图像数值的范围为[0,1], 需将其转换为[0,255]
        for current_image_id in range(images_train_count):
            for i in range(pixels_count_per_image):
                if mnist.train.images[current_image_id][i] != 0:
                    mnist.train.images[current_image_id][i] *= 255
                    print(mnist.train.images[current_image_id][i])

            if ((current_image_id + 1) % 50) == 0:
                print("已转换%d张，共需转换%d张" %
                      (current_image_id + 1, images_train_count))

        # 创建train images的保存目录, 按标签保存
        for i in range(10):
            dir = "%s/%s" % (save_dir, i)
            print(dir)
            if not os.path.exists(dir):
                os.mkdir(dir)

        # indices = [0, 0, 0, ..., 0]用来记录每个标签对应的图片数量
        indices = [0 for x in range(0, 10)]
        for i in range(images_train_count):
            new_image = Image.new("L", (cols, rows))
            # 遍历new_image 进行赋值
            for r in range(rows):
                for c in range(cols):
                    new_image.putpixel(
                        (r, c), int(mnist.train.images[i][c + r * cols]))

            # 获取第i张训练图片对应的标签
            label = labels[i]
            image_save_path = "%s/%s/%s.png" % (save_dir, label,
                                                indices[label])
            indices[label] += 1
            new_image.save(image_save_path)

            # 打印保存进度
            if ((i + 1) % 50) == 0:
                print("图片保存进度: 已保存%d张，共需保存%d张" % (i + 1, images_train_count))
    else:
        print("图片数量与标签数量不一致!")


if __name__ == '__main__':
    mnist_dir = r"C:\Users\61060\Desktop\NPU-LQ\大三下\模式识别与机器学习\工程3+2019302863+李奇\project3"
    save_dir = r"C:\Users\61060\Desktop\NPU-LQ\大三下\模式识别与机器学习\工程3+2019302863+李奇\project3\MNIST"
    extract_mnist(mnist_dir, save_dir)



