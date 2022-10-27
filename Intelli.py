#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

# initialize the data set we'll work with
url = 'https://raw.githubusercontent.com/kalleleander/Intellishore_prep/blob/8ee6b2413cae759c2205b02fbf3cd0f1d49a5f59/train_ctrUa4K.csv'
training_data = pd.read_csv(url, index_col=0)
print(training_data.head(5))

# define the model
kmeans_model = KMeans(n_clusters=2)

# assign each data point to a cluster
dbscan_result = dbscan_model.fit_predict(training_data)

# get all of the unique clusters
dbscan_clusters = unique(dbscan_result)

# plot the DBSCAN clusters
for dbscan_cluster in dbscan_clusters:
    # get data points that fall in this cluster
    index = where(dbscan_result == dbscan_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the DBSCAN plot
pyplot.show()
