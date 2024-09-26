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
import numpy as np
from sklearn.model_selection import train_test_split
X, y = make_blobs(random_state=42)
#X, X_test, y, y_test = train_test_split(X_initial, y_initial, random_state=42)
#mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
#plt.xlabel("Feature 0")
#plt.ylabel("Feature 1")
#plt.legend(["Class 0", "Class 1", "Class 2"])

#fitting the SVC linear model on the model data
linear_svc = LinearSVC(dual='auto').fit(X, y)
#dual status set to auto to suppress warning

#getting coeffients and intercept of model 
#print('shape of Coeffients is {} by {}'.\
#      format(linear_svc.coef_.shape[0], \
#             linear_svc.coef_.shape[1]))
#print('Model interecept has the shape {}'.format(linear_svc.intercept_.shape))

#at this point, there are 3 distinct groupings on the plot
#shape of coeffients is 3 by 2, shape of intercept is 3

#now watching hoe the coefficnets actually look like
#print('Coeffients in the new Linear SVC model are: {}'.format(linear_svc.coef_))
#print('Intercept value is {}'.format(linear_svc.intercept_))
    
#to understand better how the model works, 
#test the class of a random value in dataset

#printing first line of data and first line of target
#print('{:.2f} and {:.2f} are for class {}'.format(X_test[2][0], X_test[2][1], y_test[2]))

#predicting class for those features
#print('predicted class for {:.2f} and {:.2f} is {}'.format(X_test[2][0], X_test[2][1], linear_svc.predict([X_test[2]])[0]))
    
#print('Accuracy score for model is {:.2f}%'.format(100*linear_svc.score(X_test, y_test)))

#visualizing lines given by the 3 classifiers
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svc.coef_, linear_svc.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line*coef[0] + intercept)/coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel('Feature0')
plt.ylabel('Feature1')
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line Class 0', 'Line Class 1', 'Line Class 2'], loc= (1.01, 0.3))





