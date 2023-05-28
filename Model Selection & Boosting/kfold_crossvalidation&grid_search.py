# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 20:21:00 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf') #default is rbf
classifier.fit(x_train, y_train)

print(sc.inverse_transform(x_test[[1]]))
print(classifier.predict(sc.transform([[30, 87000]])))

y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred)) #confusion matrix shows how many correct and wrong predictions for each category there are
print(accuracy_score(y_test, y_pred)) #returns rate of correct predictions from 0 to 1

#k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10) #cv is the no. of folds desired, usually 10
print('Accuracy: {:.2f} %'.format(accuracies.mean() * 100)) #prints out the average accuracy percentage in 2 decimal places for all of the folds
print('Standard Deviation: {:.2f} %'.format(accuracies.std() * 100)) #prints out the standard deviation of the accuracies in 2 decimal places for all of the folds

#grid search
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [0.25, 0.5, 0.75, 1], 'kernel' : ['linear']},
              {'C' : [0.25, 0.5, 0.75, 1], 'kernel' : ['rbf'], 'gamma' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}] #creates a list of all different values of hyperparameters to be tested
grid_search = GridSearchCV(estimator=classifier, 
                           param_grid=parameters, 
                           scoring='accuracy',
                           cv=10, #each of these combinations of hyperparameters will be tested with k fold cross validation so cv is needed
                           n_jobs=-1) #uses all the computer processors for this process to speed things up

grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print('Best Accuracy: {:.2f} %'.format(best_accuracy * 100))
print('Best Parameters: ', best_parameters)