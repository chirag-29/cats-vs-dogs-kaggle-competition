
import os
import numpy as np 
import cv2 
from random import shuffle
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.core import Dropout
import pandas as pd
import matplotlib.pyplot as plt


image_size = 50
train_img = '/Users/chintanpatel/Downloads/all/train'
test_img = '/Users/chintanpatel/Downloads/all/test'

def preprocessing_train(train_img):
    training_data = []
    for img in tqdm(os.listdir(train_img)):
        label = img.split('.')[0]
        if label == 'cat':
            label = [0]
        elif label == 'dog':
            label = [1]
        path = os.path.join(train_img,img)
        img = cv2.resize(cv2.imread(path,0),(image_size,image_size))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    return training_data
def  preprocessing_test():
    test_data = []
    for img in os.listdir(test_img):
        path = os.path.join(test_img,img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path,0),(image_size,image_size))
        test_data.append([np.array(img),np.array(img_num)])
    return test_data

 
train_data = preprocessing_train(train_img)

train = train_data[:-1000]
test = train_data[-1000:]

X_train = np.array([i[0] for i in train]).reshape(-1,image_size,image_size,1)
y_train = np.array([i[1] for i in train])

X_test = np.array([i[0] for i in test]).reshape(-1,image_size,image_size,1)
y_test = np.array([i[1] for i in test])

model = Sequential()

model.add(Conv2D(32,kernel_size = 2,activation='relu',padding='same',input_shape = (image_size,image_size,1)))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(64,kernel_size = 2,activation = 'relu',padding='same'))
model.add(MaxPooling2D(pool_size =(2,2),padding='same'))
model.add(Conv2D(32,kernel_size = 2,activation = 'relu',padding='same'))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model.add(Conv2D(64,kernel_size = 2,activation = 'relu',padding='same'))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model.add(Conv2D(32,kernel_size = 2,activation = 'relu',padding='same'))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model.add(Conv2D(32,kernel_size = 2,activation = 'relu',padding='same'))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model.add(Conv2D(32,kernel_size = 2,activation = 'relu',padding='same'))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model.add(Conv2D(32,kernel_size = 2,activation = 'relu',padding='same'))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.8))

model.add(Dense(1,activation = 'sigmoid'))

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

model.fit(X_train,y_train, validation_data = (X_test,y_test),epochs =10)


test_data = preprocessing_test()
fig = plt.figure()
for num,data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(3,4,num+1)
    original = img_data
    data = img_data.reshape(1,image_size,image_size,1)
    
    y_pred = model.predict([data])
    y_pred = y_pred > 0.5
    
    if y_pred == 1:
        str_out = 'dog'
    elif y_pred == 0:
        str_out = 'cat'
    y.imshow(original,cmap='gray')
    plt.title(str_out)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False) 
plt.show()

with open('submission-file.csv','w') as f:
    f.write('id,label\n')
with open('submission-file.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        original = img_data
        data = img_data.reshape(1,image_size,image_size,1)    
        y_pred = model.predict([data])
        y_pred = y_pred > 0.5
        f.write('{},{}\n',format(img_num,y_pred))
         