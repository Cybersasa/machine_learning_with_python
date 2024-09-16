# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:34:39 2024

@author: Cybersasa

"""

# understanding supervised learning algorithms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


"""
#generating mglearn classification dataset
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
print('X Shape is {}'.format(X.shape))
print('y Shape is {}'.format(y.shape))
print(X)

#generating mglearn regression dataset
X, y = mglearn.datasets.make_wave(n_samples=40)
print(33)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")


"""

#importing Wisconsin Breast Cancer dataset that examines sizes of cancer tumors
#This is a real classification dataset with 2 classes (binary classification)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print('Keys in Cancer Dataset: {}'.format(cancer.keys()))
print('There are {} rows and {} columns in the cancer \
      dataset'.format(cancer.data.shape[0], cancer.data.shape[1]))
print('In our data, there are {} target names (classes) whose names \
      are {} and {}.'.format(cancer.target_names.shape[0], \
      cancer.target_names[0], cancer.target_names[1]))

# k-NN (k nearest neighbors) model is arguable the simplest machine learning model
#you simply store the data.
#to make a prediction using k-NN for a new data point, just check the data points
#in the training set that are closest to the new data point

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=91)
print('\nSize of training set X is {} by {} and y \
is {}.'.format(X_train.shape[0], \
      X_train.shape[1], y_train.shape[0]))
print('\nSize of testing set X is {} by {} and y \
is {}.'.format(X_test.shape[0], X_test.shape[1], y_test.shape[0]))

# training cancer dataset using the knn algorithm
cancerr = KNeighborsClassifier(n_neighbors=3)
cancerr.fit(X_train, y_train)

# after training, we test it using testing set
print('\nAccuracy of our knn model in predicting cancerous tumors \
is {:.2f}%'.format(100*(cancerr.score(X_test, y_test))))

#Testing the system by adjusting one value by a small margin and
#predicting whether the tumor is cancerous or not

from random import randint
p = randint(0, len(X_test))
p_value_X = X_test[p]
p_value_y = y_test[p]
p_value_X[0] = p_value_X[0]*1.1
print('\nThe value in position {} is {} and its class is {}'.format(p, p_value_X, p_value_y))
print('\nI predict the class of that data is {}'.format(cancerr.predict(p_value_X)))
print('\nAfter adjusting first value in data, the prediction is {}'.format(cancerr.predict(p_value_X)))



