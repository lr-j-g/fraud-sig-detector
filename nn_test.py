# -*- coding: utf-8 -*-
"""
Created on Sun May 28 16:51:56 2023

@author: Luke
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import cv2
import glob
from sklearn.utils import shuffle
import tensorflow as tf


# DATA PRE-PROCESSING

gen = [glob.glob("Dataset\\Used\\Training Set\\1\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\2\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\3\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\4\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\5\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\6\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\7\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\9\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\10\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\11\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\12\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\13\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\14\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\15\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\16\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\17\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\18\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\19\\original\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\20\\original\\*.png"),
       ]
                 
forg =[glob.glob("Dataset\\Used\\Training Set\\1\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\2\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\3\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\4\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\5\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\6\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\7\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\9\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\10\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\11\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\12\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\13\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\14\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\15\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\16\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\17\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\18\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\19\\forgery\\*.png"),
       glob.glob("Dataset\\Used\\Training Set\\20\\forgery\\*.png"),
       ]

train_data = []
train_labels = []

test_data = []
test_labels = []

for data in range(len(gen)):
    for i in gen[data]:
        if data == 3:
            image = cv2.imread(i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            test_data.append(image)
            test_labels.append(0)
        else:
            image = cv2.imread(i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            train_data.append(image)
            train_labels.append(0) #genuine = 0
        
for data in range(len(forg)):
    for j in forg[data]:
        if data == 3:
            image = cv2.imread(j)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            test_data.append(image)
            test_labels.append(1)
        else:
            image = cv2.imread(j)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            train_data.append(image)
            train_labels.append(1) #forged = 1

train_data = np.array(train_data)/255.0
train_labels = np.array(train_labels)

test_data = np.array(test_data)/255.0
test_labels = np.array(test_labels)

train_data,train_labels = shuffle(train_data,train_labels)

test_data,test_labels = shuffle(test_data,test_labels)

# NN MODEL

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=(224,224,3),activation='relu')) 
model.add(tf.keras.layers.MaxPooling2D(3,3))

model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.3))

model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
model.summary()

# TRAINING NN MODEL
progess=model.fit(train_data, train_labels, epochs=2)

pred = model.predict(test_data)
