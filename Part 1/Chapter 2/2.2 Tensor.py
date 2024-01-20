import numpy as np
x = np.array(12)
print(x)
print(x.ndim)   # 0维

x = np.array([12,3,6,14,7])
print(x)
print(x.ndim)   # 1D张量

x = np.array([[5,78,2,34,0],
              [6,79,3,35,1],
              [7,80,4,36,2]])
print(x)
print(x.ndim)   # 2D张量

x = np.array([[[5,78,2,34,0],
               [6,79,3,35,1],
               [7,80,4,36,2]],
              [[5,78,2,34,0],
               [6,79,3,35,1],
               [7,80,4,36,2]],
               [[5,78,2,34,0],
               [6,79,3,35,1],
               [7,80,4,36,2]],])
print(x)
print(x.ndim)   # 3D 张量

from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
# 8位整数组成的 3D 张量
print(train_images.ndim)    # 轴数
print(train_images.shape)   #形状
print(train_images.dtype)   #数据类型

#显示第 4 个数字
digit = train_images[4]
import matplotlib.pyplot as plt 
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()

"""Numpy中操作张量"""
my_slice = train_images[10:100] #10-99
# my_slice = train_images[10:100, :, :]
# my_slice = train_images[10:100, 0:28, 0:28]
print(my_slice.shape)

my_slice = train_images[:, 14:, 14:]    # 右下角 14 * 14 区域
my_slice = train_images[:, 7:-7, 7:-7]  # 中间 14 * 14 区域

# 小批量拆分
batch= train_images[:128]   # 小批量0
batch= train_images[128:256]   # 小批量1
n = 10
batch= train_images[128 * n:128*(n+1)]   # 小批量n














