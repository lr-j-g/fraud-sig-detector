# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:47:44 2023

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

gen = []
                 
forg =[]

#pre-load data
for i in range(21,27):
    d = "Dataset\\Used\\Test Set\\{}\\original\\*.png" 
    d = d.format(i)
    gen.append(glob.glob(d))

for i in range(21,27):
    d = "Dataset\\Used\\Test Set\\{}\\forgery\\*.png"
    d = d.format(i)
    forg.append(glob.glob(d))


#data pre-processing
test_data = []
test_labels = []

for data in range(len(gen)):
    for i in gen[data]:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        test_data.append(image)
        test_labels.append(0)
        
for data in range(len(forg)):
    for j in forg[data]:
        image = cv2.imread(j)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        test_data.append(image)
        test_labels.append(1)
        
test_data = np.array(test_data)/255.0
test_labels = np.array(test_labels)
test_data,test_labels = shuffle(test_data,test_labels)

##NN MODEL
model = tf.keras.models.load_model('sigvalidation.model') #loads trained model
loss, accuracy = model.evaluate(test_data,test_labels)

print("Loss:",loss)
print("Accuracy:",accuracy)

pred = model.predict(test_data)

print(pred)
        
