# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 05:58:55 2024

@author: Cybersasa
"""

#naive bayes (NB) classifers are faster than linear models
#NB classifiers have slightly poorer generalizations compared to linear models
#sklearn supports 3 NB classifers/ models:
#    i) GaussianNB: for continous data
#    ii) BernoulliNB: for binary data (counts non-zero data for each class)
#    iii) MultinomialNB: counts data
#BernoulliNB and MultinomialNB are used in text data classification

#demonstration of how Bernoulli works by counting non-zeros elements for
#each class
import numpy as np
X = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 1, 0]
    ])
y = np.array([0, 1, 0, 1])
counts = {}
for label in np.unique(y):
    counts[label] = X[y ==label].sum(axis=0)
print(counts)

#MultinomialNB stores the average of each feature for each class
#GaussianNB stores average and standard deviation of each feature for
#each class
#In GaussianNB and MultinomianNB, to make a prediction, the new data point 
#is compared to the statistics of each of the classes



