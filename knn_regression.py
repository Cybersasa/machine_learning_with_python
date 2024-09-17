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

X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X, y)

#making predic

