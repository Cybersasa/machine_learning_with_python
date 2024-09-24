# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:23:11 2024

@author: Cybersasa
"""

#handling multiple classes in linear classification models
#one-vs.-rest used to convert binary linear classification
#algorithms to multi-class algorithms
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import mglearn
from sklearn.svm import LinearSVC
X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])

#fitting the SVC linear model on the model data
linear_svc = LinearSVC(dual='auto').fit(X, y)
#dual status set to auto to suppress warning

#getting coeffients and intercept of model 
print('shape of Coeffients is {} by {}'.\
      format(linear_svc.coef_.shape[0], \
             linear_svc.coef_.shape[1]))
print('Model interecept has the shape {}'.format(linear_svc.intercept_.shape))

    
    
    
    
    
    