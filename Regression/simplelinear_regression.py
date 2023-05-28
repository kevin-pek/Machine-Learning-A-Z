# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:44:32 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=1)

from sklearn.linear_model import LinearRegression as lr #simple linear regression model
regressor = lr()
regressor.fit(x_train, y_train) #training the model based on the training set

y_prediction = regressor.predict(x_test) #predicting test results

plt.scatter(x_train, y_train, color='red') #plots the training data points
plt.plot(x_train, regressor.predict(x_train), color='blue') #draws the regression line of the dataset, in this case is a straight line
plt.title('Salary VS Experience (Training Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color='red') #plots the test data points
plt.plot(x_train, regressor.predict(x_train), color='blue') #doesnt matter if the train set line is used as test and train set have the same regression line
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

print(regressor.predict([[7]])) #predict salary of person with 7 years exp

print(regressor.coef_) #coefficient of independent variable
print(regressor.intercept_) #y intercept