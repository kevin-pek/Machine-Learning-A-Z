# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:40:21 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3) #\t means data separated by tab, quoting to ignore quotes in the text data

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #replaces non a-z characters with space, dataset[review][i] is the same as calling dataset.iloc[i, 0]
    review = review.lower() #changes all to lowercase
    review = review.split() #split the review into individual words
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not') #removes not from the list of stopwords
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)] #applies stemming to all words in each review, removing all stopwords
    review = ' '.join(review) #joins all the words in the review back together and adding a space in between all words
    corpus.append(review)
    
print(corpus)

from sklearn.feature_extraction.text import CountVectorizer
#print(len(x[0]))
cv = CountVectorizer(max_features=1500) #in this dataset there were 1566 total words, so we remove some of the least used words which wont provide value by setting max_features
x = cv.fit_transform(corpus).toarray() #toarray to turn matrix into 2d array
y = dataset.iloc[:, -1].values

print(x)

#using naive bayes to train the model to predict whether a review is positive or negative
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred)) #confusion matrix shows how many correct and wrong predictions for each category there are
print(accuracy_score(y_test, y_pred))