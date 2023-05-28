# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:32:04 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Ads_CTR_Optimisation.csv') #this file simulates the user clicking or not clicking on an ad

import math

N = 10000
d = 10
ads_selected = []
number_of_selections = [0] * d #creates d number of rows
sum_of_rewards = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for j in range(0, d):
        if (number_of_selections[j]>0):
            average_reward = sum_of_rewards[j] / number_of_selections[j]
            delta_i = math.sqrt(1.5 * math.log(n + 1) / number_of_selections[j])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 #ensures that within the first d rounds, every single d number of ads have been selected at least once
            
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = j
            
    ads_selected.append(ad)
    number_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] += reward
    total_reward += reward

plt.title('Histogram of Ads Selected')        
plt.hist(ads_selected)
plt.xlabel('ads')
plt.ylabel('number of times selected')
plt.show()
