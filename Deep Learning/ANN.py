# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:23:47 2020

@author: kevin
"""

import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(x)

#encoding gender column
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x[:, 2] = encoder.fit_transform(x[:, 2])

#encoding geography column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder as ohe
ct = ColumnTransformer(transformers=[('encoder', ohe(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


print(x)

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

ann = tf.keras.models.Sequential() #initialises ann as a sequence of layers
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) #relu means rectifier activation function
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) #this line is a template to add a layer to the neural network

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #since dv is binary, only one needed, if 3 possible outcomes, need 3 since dv will be encoded
#output activation function is sigmoid since sigmoid will give the probabilities of outcome being whichever value
#for categorical, activation function is 'softmax'

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #loss function must be this one for binary outcomes, categorical is 'categorical_crossentropy'
type(x_train)
print(x_train)
'''
ann.fit(x_train, y_train, batch_size=32, epochs=1000)

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]) > 0.5))

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5) #turns probabilities into either 0 or 1 based on the threshold
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
'''
