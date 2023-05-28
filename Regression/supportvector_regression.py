# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:23:50 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Python/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler #applying feature scaling
lvl_scale = StandardScaler()
x = lvl_scale.fit_transform(x)
#make sure different features have their own scaler
sal_scale = StandardScaler()
y = y.reshape(len(y), 1) #reshape y into vertical array
y = sal_scale.fit_transform(y)

from sklearn.svm import SVR #training the SVR model on the dataset
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

#predicting a result
print(sal_scale.inverse_transform(regressor.predict(lvl_scale.transform([[6.5]])))) #make sure the independent variable for the prediction is scaled to the scale

plt.scatter(lvl_scale.inverse_transform(x), sal_scale.inverse_transform(y)/1000, color='red')
plt.plot(lvl_scale.inverse_transform(x), sal_scale.inverse_transform(regressor.predict(x))/1000, color='blue')
plt.title('Salary VS Position Held (SVR)')
plt.xlabel('Level of Position Held')
plt.ylabel('Salary (in thousands)')
plt.show()