"""预测房价：回归问题 """

"""1 加载波士顿房价数据"""
from keras.datasets import boston_housing
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()
print(train_data[0])
print(train_data.shape)
print(test_data[0])
print(test_data.shape)

"""2 数据标准化"""
mean = train_data.mean(axis=0)  #各列求均值
train_data -=mean
std = train_data.std(axis=0)    #标准差
train_data /= std
test_data -= mean
test_data /= std

print(train_data[0])
print(test_data[0])

"""
3 模型定义
    网络架构: 两个中间层，每层64个隐藏单元
              第三层输出一个标量，预测房价
    中间层激活函数：relu（rectified linear unit，整流线性单元），将负值归零
    输出层激活函数：无
    将模型定义为函数，可多次实例化
"""
from keras import models
from keras import layers    

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='mse',   # 均方误差
                  metrics=['mae'])  #平均绝对误差
    return model

"""4 k折验证"""
import numpy as np
import time

k = 4
num_val_samples = len(train_data) // k  # 每折样本数
"""
num_epochs = 100    # 训练轮次
all_scores = []

tic = time.time()   # 起始时间

for i in range(k):
    print('processing fold #',i)
    # 验证集：第i个分区的数据
    val_data = train_data[i * num_val_samples:(i+1) * num_val_samples]  
    val_targets = train_targets[i * num_val_samples:(i+1) * num_val_samples]
    
    # 训练集，其他所有分区的数据
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],  
                                         train_data[(i+1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i+1) * num_val_samples:]],
                                        axis=0)    
    model = build_model() # 构建Kares模型（已编译）
    model.fit(partial_train_data,
              partial_train_targets,
              epochs=num_epochs,
              batch_size=1,
              verbose=0)    # 静默模式
    val_mse,val_mae = model.evaluate(val_data,val_targets,verbose=0)    # 在验证集上评估
    all_scores.append(val_mae)
    
toc = time.time()   # 结束时间
print("time used:", toc - tic)  #训练用时

print(all_scores)
print(np.mean(all_scores))
"""
"""5 保存每折的验证结果"""
num_epochs = 500    #训练轮次
all_mae_histories = []

tic = time.time()   # 起始时间
for i in range(k):
    print('processing fold #',i)
    # 验证集：第i个分区的数据
    val_data = train_data[i * num_val_samples:(i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i+1) * num_val_samples]
    
    # 训练集，其他所有分区的数据
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i+1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                         train_targets[(i+1) * num_val_samples:]],
                                        axis=0)    
    model = build_model() # 构建Kares模型（已编译）
    history = model.fit(partial_train_data,
                        partial_train_targets,
                        validation_data=(val_data,val_targets),
                        epochs=num_epochs,
                        batch_size=1,
                        verbose=0)
    
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
   
toc = time.time()   # 结束时间
print("time used:", toc - tic)  #训练用时

"""6 计算所有轮次中的K折验证分数的平均值"""
average_mae_history = [np.mean([x[i]for x in all_mae_histories])for i in range(num_epochs)]
print(average_mae_history)

"""7 绘制验证分数"""
import matplotlib.pyplot as plt

plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

"""
8 重新绘制验证分数：
    删除前10个数据点，因为取值范围与其他不同
    每个数据点替换为前面数据点的指数移动平均值，得到光滑的曲线
"""
def smooth_curve(points,factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
    
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1,len(smooth_mae_history) + 1),smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

"""9 训练最终模型"""
model = build_model()
model.fit(train_data,
          train_targets,
          epochs=80,
          batch_size=16,
          verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data,test_targets)
print(test_mse_score)
print(test_mae_score)