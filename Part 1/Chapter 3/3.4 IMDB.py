"""
3.4 电影评价分类：二分类问题
根据电影评论的文字内容将其划分为正面或负面
"""

"""1 加载IMBD数据集"""
from msilib import sequence
from keras.datasets import imdb

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)
print(train_data[0])
print(test_labels.shape)

print(max([max(sequence) for sequence in train_data] )) # 最大单词索引

# 将评论解码为英文单词
word_index = imdb.get_word_index()  # 将单词映射为整数索引的字典
reverse_word_index = dict([(value,key)for (key,value) in word_index.items()])   #键值颠倒，将整数索引映射为单词
decoded_review = ' '.join([reverse_word_index.get(i-3,'?')for i in train_data[0]])  #将评论解码，0，1，2为保留索引
print(decoded_review)

"""2 准备数据"""
#将整数序列编码为二进制矩阵
import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))  # 创建形状为(len(sequences),dimension)的零矩阵
    for i,sequence in enumerate(sequences): 
        results[i,sequence] = 1 # 将 results[i]的指定索引设置为1
    return results
x_train = vectorize_sequences(train_data)   # 将训练数据向量化
x_test = vectorize_sequences(test_data)     # 将测试数据向量化

y_train = np.array(train_labels).astype('float32')  # 训练标签向量化
y_test = np.array(test_labels).astype('float32')    # 测试标签向量化

print(x_train[0])
print(y_train[0])
print(x_test[0])
print(y_test[0])

"""
3 模型定义

    网络架构: 两个中间层，每层16个隐藏单元
              第三层输出一个标量，预测当前评论的感情
    中间层激活函数：relu（rectified linear unit，整流线性单元），将负值归零
    输出层激活函数：sigmoid，将任意值压缩到[0, 1]区间，作为概率输出
"""
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

"""4.1 编译模型"""
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
"""
# 4.2 配置优化器
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4.3 自定义损失函数和指标
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

"""

"""5 划分验证集"""
x_val = x_train[:10000] # 特征验证集
partial_x_train = x_train[10000:]

y_val = y_train[:10000] # 对应的标签验证集
partial_y_train = y_train[10000:]

"""6 训练模型"""
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))
history_dict = history.history
print(history_dict.keys())

"""7 绘制训练损失和经验损失"""
import matplotlib.pyplot as plt

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""8 绘制训练精度和验证精度"""
plt.clf()   # 清空图像
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs,acc_values,'bo',label='Training acc')
plt.plot(epochs,val_acc_values,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""9 从头开始重新训练一个模型,训练4轮"""

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results=model.evaluate(x_test,y_test)
print(results)

# 预测结果
y_pred=model.predict(x_test)
print(y_pred)

"""
进一步实验：
    使用一个或者三个隐藏层；
    使用更多或更少的隐藏单元，例如32个、64个等；
    使用mse损失函数代替binary_crossentropy；
    使用tanh激活代替relu

"""