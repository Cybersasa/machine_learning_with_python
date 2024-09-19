# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:19:08 2024

@author: Cybersasa
"""

#learning about linear models for classification
#starting with binary classification models for simplicity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import mglearn
from matplotlib import pyplot as plt

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10,3))