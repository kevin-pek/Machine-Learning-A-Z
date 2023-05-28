# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:25:08 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Ads_CTR_Optimisation.csv') #this file simulates the user clicking or not clicking on an ad

import math, random

N = 500
d = 10
ads_selected = []
number_of_rewards_1 = [0] * d #creates d number of rows
number_of_rewards_0 = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random_draw=0
    for j in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[j] + 1, number_of_rewards_0[j] + 1)
            
        if random_beta > max_random_draw:
            max_random_draw = random_beta
            ad = j
            
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward==1:
        number_of_rewards_1[ad] += 1
    else:
        number_of_rewards_0[ad] += 1
    total_reward += reward

plt.title('Histogram of Ads Selected')        
plt.hist(ads_selected)
plt.xlabel('ads')
plt.ylabel('number of times selected')
plt.show()
