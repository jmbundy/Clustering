# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 13:06:08 2017

@author: jmbun
"""

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

df = pd.read_excel('C:\\Users\\jmbun\\OneDrive - TABS Analytics\\Clients\\5 Hour Energy\\Elasticity\\Clustering.xlsx')
array=df.values
X=array[:,1:4]

clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(X)
    clusassign=model.predict(X)
    meandist.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1))
    / X.shape[0])

model.labels=array[:,0]
plt.plot(clusters, meandist)
plt.xlabel('Number of Clusters')
plt.ylabel('Average Distance')
plt.title('Selection of k Clusters')

# Interpret 3 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(X)
clusassign3=model3.predict(X)
centroids3=model3.cluster_centers_
# plot clusters

plt.scatter(centroids3[:,0], centroids3[:,1], s=99, linewidths=3, zorder=10)
labels=model.labels

# Interpret 5 cluster solution
model5=KMeans(n_clusters=5)
model5.fit(X)
clusassign5=model5.predict(X)
centroids5=model5.cluster_centers_
# plot clusters

plt.scatter(centroids5[:,0], centroids5[:,1], s=99, linewidths=3, zorder=10)
labels=model.labels

kout=pd.DataFrame({"GEO": labels,"k=3": clusassign3, "k=5": clusassign5 })

kout.to_csv('C:\\Users\\jmbun\\OneDrive - TABS Analytics\\Clients\\5 Hour Energy\\Elasticity\\ClustersOut.csv', index=False)

