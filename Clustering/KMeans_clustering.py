# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 00:06:34 2020

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [2, 3, 4]].values #y is not needed since there is no dependent variable in the dataset

#print(x)

#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#x = np.array(ct.fit_transform(x))

print(x)

from sklearn.cluster import KMeans
wcss = [] #list of wcss values for each instance of kmeans, each having different no. of clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) #intertia_ gives the wcss value
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no. of clusters')
plt.ylabel('WCSS')
plt.show()

#assuming from the graph, we decided that 5 clusters is the best option
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x) #trains the model on the dataset and returns the values of the dependent variable, which have 5 different values in this case
print(y_kmeans)


from mpl_toolkits.mplot3d import Axes3D #for plotting in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[y_kmeans==0, 0], x[y_kmeans==0, 1], x[y_kmeans==0, 2], c='red', label='Cluster 1', s=20)
ax.scatter(x[y_kmeans==1, 0], x[y_kmeans==1, 1], x[y_kmeans==1, 2], c='blue', label='Cluster 2', s=20)
ax.scatter(x[y_kmeans==2, 0], x[y_kmeans==2, 1], x[y_kmeans==2, 2], c='green', label='Cluster 3', s=20)
ax.scatter(x[y_kmeans==3, 0], x[y_kmeans==3, 1], x[y_kmeans==3, 2], c='yellow', label='Cluster 4', s=20)
ax.scatter(x[y_kmeans==4, 0], x[y_kmeans==4, 1], x[y_kmeans==4, 2], c='purple', label='Cluster 5', s=20)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=50, c='black', label='Centroids')

plt.title('KMeans Clustering')
ax.set_xlabel('Age')
ax.set_ylabel('Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.legend()
plt.show()