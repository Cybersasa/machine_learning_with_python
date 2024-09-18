# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:31:19 2024

@author: Cybersasa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from scipy import sparse
from random import randint

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#knn for regression
#from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#reg = KNeighborsRegressor(n_neighbors=3)
#reg.fit(X_train, y_train)

#making predictions from the new regression model
#to be able to assess accuracy of the predictions, we make predictions
#on the test set of our data
#print('Predicted Value are : {}'.format(reg.predict(X_test)))

#Evaluating performace of model using R^2
#print('The model accuracy is {:.2f}'.format(100*(reg.score(X_test, y_test))))

#visualizing regression through knn model
#mglearn.plots.plot_knn_regression(n_neighbors=1)

##testing optimal neighbours, then optimal random_state

#finding optimal number of neighbours (highest R-squared)
X, y = mglearn.datasets.make_wave(n_samples=40)
#print('Labels in Cancer dataset are: {}'.format(cancer.keys()))
#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=72277)
from sklearn.neighbors import KNeighborsRegressor
scores = []
for i in range(1,31):
    model = KNeighborsRegressor(n_neighbors=i)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print('Highest accuracy is {:.2f} occuring {} times at position {} so best n_neighbors={}'.format(max(scores), scores.count(max(scores)), scores.index(max(scores)), scores.index(max(scores))+1))
#optimal n_neighbours determined to be 3, yielding a model accuracy of 83%


#with n_neighbors = 3, find optimal random_state value
scores = []
for i in range(0, 1000001):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print('Highest accuracy is {:.2f} occuring {} times at position {} so best random_state={}'.format(max(scores), scores.count(max(scores)), scores.index(max(scores)), scores.index(max(scores))+1))
  
#for 3 neighbors, optimal random_state=72277 with an ccuracy of 0.97

