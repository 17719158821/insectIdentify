import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

# data_dir = './cleaned/'
data_dir = 'dataset'
# 下面都是加载label
Name = []
for file in os.listdir(data_dir):
    Name += [file]

print(Name)
print(len(Name))

N = []
for i in range(len(Name)):
    N += [i]

normal_mapping = dict(zip(Name, N))
reverse_mapping = dict(zip(N, Name))


def mapper(value):
    return reverse_mapping[value]


dataset = []
count = 0
for file in os.listdir(data_dir):
    path = os.path.join(data_dir, file)
    for im in os.listdir(path):
        image = load_img(os.path.join(path, im), grayscale=False, color_mode='rgb', target_size=(200, 200))
        image = img_to_array(image)
        image = image / 255.0
        dataset += [[image, count]]
    count = count + 1

n = len(dataset)
print(n)

num = []
for i in range(n):
    num += [i]
random.shuffle(num)
print(num[0:5])

data, labels = zip(*dataset)
data = np.array(data)
labels = np.array(labels)

train = data[num[0:(n // 10) * 8]]
trainlabel = labels[num[0:(n // 10) * 8]]

test = data[num[(n // 10) * 8:]]
testlabel = labels[num[(n // 10) * 8:]]
# '''
# 样本数量大于20后打开
train = data[num[0:(n // 10) * 8]]
trainlabel = labels[num[0:(n // 10) * 8]]

test = data[num[(n // 10) * 8:]]
testlabel = labels[num[(n // 10) * 8:]]
# '''

# keras将类别向量转换为二进制（只有0和1）的矩阵类型表示
trainlabel2 = to_categorical(trainlabel)

# 分割测试集与训练集， test_size=0.2,测试数据占比20%
trainx, testx, trainy, testy = train_test_split(train, trainlabel2, test_size=0.2, random_state=44)

# print(trainx.shape)
# print(testx.shape)
# print(trainy.shape)
# print(testy.shape)

# 图片生成器，可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力
datagen = ImageDataGenerator(horizontal_flip=True,  # 水平反转
                             vertical_flip=True,  # 垂直反转
                             rotation_range=20,  # 旋转范围
                             zoom_range=0.2,  # 缩放范围
                             width_shift_range=0.2,  # 水平平移范围
                             height_shift_range=0.2,  # 垂直平移范围
                             shear_range=0.1,  # 透视变换的范围
                             fill_mode="nearest")  # 填充模式

# 拥有较深层数的卷积神经网络
pretrained_model3 = tf.keras.applications.DenseNet201(input_shape=(200, 200, 3), include_top=False, weights='imagenet',
                                                      pooling='avg')
pretrained_model3.trainable = False

inputs3 = pretrained_model3.input
x3 = tf.keras.layers.Dense(512, activation='relu')(pretrained_model3.output)

# 当修改分类类别数量时，修改下面的Dense里第一个参数为类别数量
outputs3 = tf.keras.layers.Dense(99, activation='softmax')(x3)

# outputs4 = tf.keras.layers.Dense(6, activation='softmax')(x3)
model = tf.keras.Model(inputs=inputs3, outputs=outputs3)
# 加载优化器 编译模型
model.compile(optimizer='NAdam', loss='categorical_crossentropy', metrics=['accuracy'])
# 设置保存checkpoint的方法，按最高分算


filepath = 'weights.best.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]

his = model.fit(datagen.flow(trainx, trainy, batch_size=32), validation_data=(testx, testy), epochs=300,
                callbacks=callback_list)
# his=model.fit(datagen.flow(trainx,trainy,batch_size=4),epochs=30,callbacks=callback_list)

# 随机读取一张照片进行推理测试
y_pred = model.predict(testx)
pred = np.argmax(y_pred, axis=1)
ground = np.argmax(testy, axis=1)
print(classification_report(ground, pred))
# model.save('my_model.h5')
get_acc = his.history['accuracy']
value_acc = his.history['val_accuracy']
get_loss = his.history['loss']
validation_loss = his.history['val_loss']

# 打印训练曲线 loss和accu
epochs = range(len(get_acc))
plt.plot(epochs, get_acc, 'r', label='Accuracy')
# plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
plt.title('Accuracy Curve')
plt.legend(loc=0)
plt.figure()
plt.show()

epochs = range(len(get_loss))
plt.plot(epochs, get_loss, 'r', label='Loss')
# plt.plot(epochs, validation_loss, 'b', label='Loss of Validation data')
plt.title('Loss Curve')
plt.legend(loc=0)
plt.figure()
plt.show()

model = load_model('weights.best.h5')
image = load_img("./1.jpg", target_size=(200, 200))

image = img_to_array(image)
image = image / 255.0
prediction_image = np.array(image)
prediction_image = np.expand_dims(image, axis=0)

prediction = model.predict(prediction_image)
value = np.argmax(prediction)
move_name = mapper(value)
print("Prediction is {}.".format(move_name))

print(test.shape)
pred2 = model.predict(test)
print(pred2.shape)

PRED = []
for item in pred2:
    value2 = np.argmax(item)
    PRED += [value2]

ANS = testlabel

accuracy = accuracy_score(ANS, PRED)
print(accuracy)
