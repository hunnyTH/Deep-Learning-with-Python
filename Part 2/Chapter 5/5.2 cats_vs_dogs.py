"""
2.24.6.16
"""
import os, shutil

original_dataset_dir = 'D:\Desktop\github\Deep-Learning-with-Python\Part 2\Chapter 5\PetImages'
base_dir = 'D:\Desktop\github\Deep-Learning-with-Python\Part 2\Chapter 5\cats_and_dogs_small'
# os.mkdir(base_dir)

# 训练集
train_dir = os.path.join(base_dir,"train")
#os.mkdir(train_dir)
# 验证集
validation_dir = os.path.join(base_dir,"validation")
#os.mkdir(validation_dir)
# 测试集
test_dir = os.path.join(base_dir,"test")
#os.mkdir(test_dir)

# 猫训练集目录
train_cats_dir = os.path.join(train_dir,"cats")
#os.mkdir(train_cats_dir)
# 猫验证集目录
validation_cats_dir = os.path.join(validation_dir,"cats")
#os.mkdir(validation_cats_dir)
# 猫测试集目录
test_cats_dir = os.path.join(test_dir,"cats")
#os.mkdir(test_cats_dir)

# 狗训练集目录
train_dogs_dir = os.path.join(train_dir,"dogs")
#os.mkdir(train_dogs_dir)
# 狗验证集目录
validation_dogs_dir = os.path.join(validation_dir,"dogs")
#os.mkdir(validation_dogs_dir)
# 狗测试集目录
test_dogs_dir = os.path.join(test_dir,"dogs")
#os.mkdir(test_dogs_dir)

"""
# 拷贝图片
fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+"\cat",fname)
    dst = os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+"\cat",fname)
    dst = os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+"\cat",fname)
    dst = os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)


fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+"\dog",fname)
    dst = os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+"\dog",fname)
    dst = os.path.join(validation_dogs_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir+"\dog",fname)
    dst = os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)
"""

print("total training cat images:",len(os.listdir(train_cats_dir)))
print("total training dog images:",len(os.listdir(train_dogs_dir)))
print("total validation cat images:",len(os.listdir(validation_cats_dir)))
print("total validation dog images:",len(os.listdir(validation_dogs_dir)))
print("total test cat images:",len(os.listdir(test_cats_dir)))
print("total test dog images:",len(os.listdir(test_dogs_dir)))

# 构建网络
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))

print(model.summary())

from keras import optimizers
model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])

# 数据预处理
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(150,150),
                                                              batch_size=20,
                                                              class_mode='binary')

# 生成器输出
for data_batch, labels_batch in train_generator:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:",labels_batch.shape)
    break

# 训练模型
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)

# 保存模型
model.save("cats_and_dogs_small_1.h5")

# 绘制
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo', label = "Training acc")
plt.plot(epochs,val_acc,'b', label = "Validation acc")
plt.title("Training and validation accuracy")
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo', label = "Training loss")
plt.plot(epochs,val_loss,'b', label = "Validation loss")
plt.title("Training and validation loss")
plt.legend()

plt.show()
