# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:08:31 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 9 - Random Forest Regression/Python/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y/1000, color='red')
plt.plot(x_grid, regressor.predict(x_grid)/1000, color='blue')
plt.title('Salary VS Position Held (Decision Tree Regression')
plt.xlabel('Level of Position Held')
plt.ylabel('Salary (in thousands)')
plt.show()

from sklearn.metrics import r2_score
r2_score(y, regressor.predict(x))