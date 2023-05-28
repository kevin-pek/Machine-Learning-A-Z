# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:37:05 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=4) #degree refers to the n value, the largest power in the polynomial linear regression
x_poly = pr.fit_transform(x) #data is not split into test and train set to give the model more data
lr2 = LinearRegression()
lr2.fit(x_poly, y)

#visualise polynomial linear regression model results
plt.scatter(x, y/1000, color='red')
plt.plot(x, lr2.predict(x_poly)/1000, color='blue')
plt.title('Salary VS Position Held')
plt.xlabel('Level of Position Held')
plt.ylabel('Salary (in thousands)')
plt.show()

#visualise SMOOTH polynomial linear regression model results
x_grid = np.arange(min(x), max(x), 0.1) #smooth the curve for the graph
x_grid = x_grid.reshape((len(x_grid), 1)) #usually not necessary to smoothen curve since there usually is a lot more data sets
plt.scatter(x, y/1000, color='red')
plt.plot(x_grid, lr2.predict(pr.fit_transform(x_grid))/1000, color='blue')
plt.title('Salary VS Position Held')
plt.xlabel('Level of Position Held')
plt.ylabel('Salary (in thousands)')
plt.show()

#predict polynomial regression results
print(lr2.predict(pr.fit_transform([[6.5]])))