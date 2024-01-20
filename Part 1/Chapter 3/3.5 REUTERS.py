"""3.5 新闻分类：多分类问题"""

"""1 加载路透社数据集"""
from cgi import test
from keras.datasets import reuters

(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)
print(train_data[0])
print(train_data.shape)
print(test_data[0])
print(test_data.shape)

# 将索引解码为新闻文本
word_index = reuters.get_word_index()  # 将单词映射为整数索引的字典
reverse_word_index = dict([(value,key)for (key,value) in word_index.items()])   #键值颠倒，将整数索引映射为单词
decoded_review = ' '.join([reverse_word_index.get(i-3,'?')for i in train_data[0]])  #将评论解码，0，1，2为保留索引
print(decoded_review)

"""2 编码数据"""
import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results

x_train = vectorize_sequences(train_data)   # 训练数据向量化
x_test = vectorize_sequences(test_data)     # 测试数据向量化

# 标签向量化
#方法一：将标签列转化为整数张量，或者使用one-hot编码（分类编码）
def to_one_hot(labels,dimension=46):
    results = np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label] = 1
    return results
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

"""
#方法二：Keras内置方法
from keras.utils import to_categorical

one_hot_train_label = to_categorical(train_labels)
one_hot_test_label = to_categorical(test_labels)
"""

"""
3 模型定义

    网络架构: 两个中间层，每层64个隐藏单元
              输出层，46个单元
    中间层激活函数：relu（rectified linear unit，整流线性单元），将负值归零
    输出层激活函数：softmax，概率分布
"""
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

"""4 编译模型"""
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"""5 划分验证集"""
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

"""6 训练模型"""
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))

"""7 绘制训练损失和经验损失"""
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(val_loss)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""8 绘制训练精度和验证精度"""
plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


"""9 从头开始重新训练一个模型，训练9轮"""
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val,y_val))
results = model.evaluate(x_test,one_hot_test_labels)
print(results)

"""完全随机精度"""
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print(float(np.sum(hits_array)/len(test_labels)))

"""10 预测结果"""
predictions = model.predict(x_test)

print(predictions[0].shape)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))
y_pred=np.zeros(len(predictions))

for i in range(len(predictions)):
    y_pred[i] = np.argmax(predictions[i])
print(y_pred)


"""
# 将标签转换为整数张量
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# 改变损失函数选择
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
"""


# 中间层维度足够大的重要性
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(4,activation='relu'))    # 中间层只有4个隐藏单元
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val,y_val))
          
"""
进一步实验:
    使用更少或更多的隐藏层单元，比如32个、128个
    使用一个或三个隐藏层

"""