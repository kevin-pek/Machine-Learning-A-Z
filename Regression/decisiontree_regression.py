# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:23:41 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 8 - Decision Tree Regression/Python/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

print(regressor.predict([[6.5]]))

X_grid = np.arange(min(x), max(x), 0.01) #used to visualise in 2d, decision tree should have at least 2 feature and will be visualised in 3d
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y/1000, color='red')
plt.plot(X_grid, regressor.predict(X_grid)/1000, color='blue')
plt.title('Salary VS Position Held (Decision Tree Regression')
plt.xlabel('Level of Position Held')
plt.ylabel('Salary (in thousands)')
plt.show()

from sklearn.metrics import r2_score
r2_score(y, regressor.predict(x))