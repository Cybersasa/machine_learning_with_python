# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:43:28 2024

@author: Cybersasa
"""
import mglearn
from sklearn.model_selection import train_test_split
# first observe the plot and then learn how to generate it
mglearn.plots.plot_linear_regression_wave()

from sklearn.linear_model import LinearRegression
X,y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#train the linear regressin model lr
lr = LinearRegression().fit(X_train, y_train)

#Understanding what is in the lr model
print(dir(lr))
print('The coeeficient is {} and y-intercept is {}.'.format(lr.coef_, lr.intercept_))
print('Equation of the straight line is y = {:.2f}x + {:.2f}'.format(lr.coef_[0], lr.intercept_))

#evaluating performance of trainign and testing sets
print('Training performance: {:.2f}'.format(lr.score(X_train, y_train)))
print('Training performance: {:.2f}'.format(lr.score(X_test, y_test)))