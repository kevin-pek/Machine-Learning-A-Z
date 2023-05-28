# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 12:25:13 2020

@author: kevin
"""

#!pip install apyori
#rmbr to install^ if not yet

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)]) #since transaction has max 20 items, loops through each row 20 times to add each item in
    
print(transactions)

from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
#support chosen assuming we want item to be bought at least 3 times per day
#confidence is adjusted based on results u want to see, started from 0.8 then keep decreasing
#lift from exp best is 3
#min max length is based on business goal, in this case u want buy 1 get 1 free so 2

results = list(rules)

print(results)

#reorganise the data into a pandas dataframe
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

print(resultsinDataFrame)

print(resultsinDataFrame.nlargest(n=10, columns='Lift'))