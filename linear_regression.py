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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#train the linear regressin model lr
lr = LinearRegression().fit(X_train, y_train)

#Understanding what is in the lr model
print(dir(lr))
print('The coeeficient is {} and y-intercept is {}.'.format(lr.coef_, lr.intercept_))
print('Equation of the straight line is y = {:.2f}x + {:.2f}'.format(lr.coef_[0], lr.intercept_))

#evaluating performance of trainign and testing sets
print('Training performance: {:.2f}'.format(lr.score(X_train, y_train)))
print('Training performance: {:.2f}'.format(lr.score(X_test, y_test)))
# that is enough with that random makewave dataset

#evaluating linear regression model through breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
#print('Variable names: {}'.format(cancer.keys()))
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
cancerr = LinearRegression().fit(X_train, y_train)
print('\nCancer Data Training Perfomance: {:.2f}'.format(cancerr.score(X_train, y_train)))
print('Cancer Data Testing Perfomance: {:.2f}'.format(cancerr.score(X_test, y_test)))
print('Size of data is {} by {}.'.format(cancer.data.shape[0], cancer.data.shape[1]))

#that was the standard linear regression algorithm. Score not so good

#exploring an alternative to standard linear regression
#alternative is ridge regression

