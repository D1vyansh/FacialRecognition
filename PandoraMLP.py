# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:40:12 2020

@author: Atharva
"""

import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
# from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
# from keras.callbacks import ReduceLROnPlateau

import tensorflow
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model

with open('./PandoraFinalLandmarks.csv', mode='r') as csvfile:
    x = list(csv.reader(csvfile))
x = np.array(x, dtype=np.float)
print(x.shape)
with open('./PandoraFinalGT.csv', mode='r') as csvfile:
    gt = list(csv.reader(csvfile))
y = np.array(gt, dtype=np.float)
print(y.shape)


yaw, pitch, roll = y[:, 0], y[:, 1], y[:, 2]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)


std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_val = std.transform(x_val)
x_test = std.transform(x_test)


BATCH_SIZE = 32
EPOCHS = 300


model = Sequential()
model.add(Dense(units=1518, activation='relu',input_dim=x.shape[1]))
#model.add(Dense(units=10000, activation='relu', kernel_regularizer='l2', input_dim=x.shape[1]))
#model.add(Dense(units=5000, activation='relu', kernel_regularizer='l2'))
#model.add(Dense(units=500, activation='relu', kernel_regularizer='l2'))
#model.add(Dense(units=100, activation='relu', kernel_regularizer='l2'))
model.add(Dense(units=3, activation='linear'))

print(model.summary())

#callback_list = [EarlyStopping(monitor='val_loss', patience=25)]
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',factor = 0.005,patience = 20,verbose = 1)

model.compile(optimizer='adam', loss='mean_squared_error')
hist = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[reduce_lr])
model.save('./Final1panmodel.h5')

print()
print('Train loss:', model.evaluate(x_train, y_train, verbose=0))
print('  Val loss:', model.evaluate(x_val, y_val, verbose=0))
print(' Test loss:', model.evaluate(x_test, y_test, verbose=0))


history = hist.history
loss_train = history['loss']
loss_val = history['val_loss']

#plt.figure()
#plt.plot(loss_train, label='train')
#plt.plot(loss_val, label='val_loss', color='red')
#plt.legend()

#model = load_model('./Mypanmodel.h5')
y_pred = model.predict(x_test)
print(len(y_pred))
print(y_pred)
print(len(y_test))
print(y_test)


pred = model.predict(x_test)
diff = abs(y_test - pred)
diff_yaw = diff[:, 0]
diff_pitch = diff[:, 1]
diff_roll = diff[:, 2]
print('Average Yaw Error: ')
print(sum(abs(diff_yaw))/len(y_test))
print('Average Pitch Error: ')
print(sum(abs(diff_pitch))/len(y_test))
print('Average Roll Error: ')
print(sum(abs(diff_roll))/len(y_test))
