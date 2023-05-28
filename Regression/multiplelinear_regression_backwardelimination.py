# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:41:53 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder as ohe
ct = ColumnTransformer(transformers=[('encoder', ohe(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0)

print(x)

from sklearn.linear_model import LinearRegression #same class as simple linear regression
regressor = LinearRegression()
regressor.fit(x_train, y_train) #trains model based on training set

y_prediction = regressor.predict(x_test)
np.set_printoptions(precision=2) #numbers rounded to 2 dp
print(np.concatenate((y_prediction.reshape(len(y_prediction), 1), y_test.reshape(len(y_test), 1)), 1)) #compare predicted results with actual results, also reshapes horizontal vectors into vertical vectors

#scikit library automatically handles dummy variable trap and backward elimination so theres no need to manually do it in code



print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]])) #predicting the profit of a Californian startup which spent 160000 in R&D, 130000 in Administration and 300000 in Marketing
print(regressor.coef_) #coefficients of independent variable
print(regressor.intercept_) #y intercept