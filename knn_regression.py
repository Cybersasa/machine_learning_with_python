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
from sklearn.neighbors import KNeighborsRegressor

#X, y = mglearn.datasets.make_wave(n_samples=40)

'''
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X, y)

#making predictions from the new regression model
#to be able to assess accuracy of the predictions, we make predictions
#on the test set of our data
#print('Predicted Value are : {}'.format(reg.predict(X_test)))

#comparing predicted values vs actual values
pred = reg.predict(X_test)
for i in range(1, len(pred+1)):
    pred
    print('{} vs {}'.format(pred[i], y_test[i]))
'''

mglearn.plots.plot_knn_regression(n_neighbors=1)
