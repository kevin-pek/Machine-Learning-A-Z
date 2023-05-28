# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 01:25:47 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [2, 3, 4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward')) #ward method minimises the variance in each of the clusters from HC

plt.title('Dendrogram')
plt.xlabel('Customer')
plt.ylabel('Euclidean Dist.')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward') #affinity is the type of distance used, in this case euclidean
y_hc = hc.fit_predict(x)

print(y_hc)

from mpl_toolkits.mplot3d import Axes3D #for plotting in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[y_hc==0, 0], x[y_hc==0, 1], x[y_hc==0, 2], c='red', label='Cluster 1', s=20)
ax.scatter(x[y_hc==1, 0], x[y_hc==1, 1], x[y_hc==1, 2], c='blue', label='Cluster 2', s=20)
ax.scatter(x[y_hc==2, 0], x[y_hc==2, 1], x[y_hc==2, 2], c='green', label='Cluster 3', s=20)
ax.scatter(x[y_hc==3, 0], x[y_hc==3, 1], x[y_hc==3, 2], c='yellow', label='Cluster 4', s=20)
ax.scatter(x[y_hc==4, 0], x[y_hc==4, 1], x[y_hc==4, 2], c='purple', label='Cluster 5', s=20)

plt.title('KMeans Clustering')
ax.set_xlabel('Age')
ax.set_ylabel('Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.legend()
plt.show()