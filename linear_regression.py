# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:43:28 2024

@author: Cybersasa
"""
#linear refression model starts here
print()
'''
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
#good for controlling complexity, therefore reducing overfitting
-Ridge regression uses coefficients like standard linear 
regression(ordinary least squares)
-ridge regession works by minimizing the value of each 
coefficient, so that each coefficient is close to 0 as much 
as possible, a process called regularization
-regularization used here is L2 regularization
-basically, regularization is explicitly restricting a model
to avoid overfitting
-ridge regression is implemented through linear_model.Ridge

'''
#testing ridge regression on cancer, diabetes, california_housing datasets
#importing the 3 datasets
from sklearn.datasets import \
    load_breast_cancer, load_diabetes, fetch_california_housing
from sklearn.linear_model import Ridge, LinearRegression

from sklearn.model_selection import train_test_split
"""
#implementing ridge regression on cancer dataset
cancer = load_breast_cancer()
print('Keys:{}'.format(cancer.keys()))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)
cancerr = Ridge().fit(X_train, y_train)
cancerrr = LinearRegression().fit(X_train, y_train)
#getting score of trainign and testing sets
print('Training Score for Cancer data (Ridge): {:.2f}'.format(cancerr.score(X_train, y_train)))
print('Testing Score for Cancer data (Ridge): {:.2f}'.format(cancerr.score(X_test, y_test)))
print('\nTraining Score for Cancer data (Linear R.): {:.2f}'.format(cancerrr.score(X_train, y_train)))
print('Testing Score for Cancer data (Linear R.): {:.2f}'.format(cancerrr.score(X_test, y_test)))

#Ridge Performed better that ODE (linear regression) in cancer dataset
#Now testing with diabetes dataset
diabetes = load_diabetes()
#print('Keys in Diabetes dataset: {}'.format(diabetes.keys()))
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=100)
diabetess = Ridge().fit(X_train, y_train)
diabetesss = LinearRegression().fit(X_train, y_train)
#getting score of trainign and testing sets
print('Training Score for Diabetes data (Ridge): {:.2f}'.format(diabetess.score(X_train, y_train)))
print('Testing Score for Diabetes data (Ridge): {:.2f}'.format(diabetess.score(X_test, y_test)))
print('\nTraining Score for Diabetes data (Linear R.): {:.2f}'.format(diabetesss.score(X_train, y_train)))
print('Testing Score for Diabetes data (Linear R.): {:.2f}'.format(diabetesss.score(X_test, y_test)))

#diabetes dataset performs very poorly
#Investigating number of parameters in the dataset
print('Diabetes dataste has {} rows and {} columns.'.format(diabetes.data.shape[0], diabetes.data.shape[1]))
print(diabetes.target[0:5])
"""

#ODE Performed better that ridge regression in diabetes dataset
#Now testing with California hosing dataset
housing = fetch_california_housing()
print('Keys in Housing dataset: {}'.format(housing.keys()))
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, random_state=0)

diabetes = load_diabetes()
#diabetes =fetch_california_housing()
#diabetes = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=100)

housingg = Ridge(0.3).fit(X_train, y_train)
housinggg = LinearRegression().fit(X_train, y_train)
#getting score of trainign and testing sets
print('Training Score for Diabetes data (Ridge): {:.2f}'.format(housingg.score(X_train, y_train)))
print('Testing Score for Diabetes data (Ridge): {:.2f}'.format(housingg.score(X_test, y_test)))
print('\nTraining Score for Diabetes data (Linear R.): {:.2f}'.format(housinggg.score(X_train, y_train)))
print('Testing Score for Diabetes data (Linear R.): {:.2f}'.format(housinggg.score(X_test, y_test)))

#In this dataset, there are exactly the same results for 
#both ODE and ridge regression

#printing the coefficients of the latest ridge model
#print('The {} coeffients in last Ridge regression Model are:\n{}'.format(len(housingg.coef_), housingg.coef_))
#counting coefficients in ODE for the same latest dataset
print('There are {} coeffients in last Simple linear regression Model.'.format(len(housinggg.coef_)))
#counting the number of columns in the latest dataset
print('There are {} columns in the dataset.'.format(housing.data.shape[1]))
#There are 8 coefficients for each model, and dataset also has 8 
#columns. Number of coefficients is therefrore the number 0f x variables in data
#printing the linear equation from Ridge regression
#print('The linear equation based on the Ridge coefficients is \
#y = {}x1 + {}x2 + {}x3 + {}x4 + {}x5 + {}x6 + {}x7 + {}x8 + {} \
#'.format(housingg.coef_[0], housingg.coef_[1], housingg.coef_[2], housingg.coef_[3], housingg.coef_[4], housingg.coef_[5], housingg.coef_[6], housingg.coef_[7], housingg.intercept_))

#printing the linear equation from ODE
#print('The linear equation based on the ODE coefficients is \
#y = {}x1 + {}x2 + {}x3 + {}x4 + {}x5 + {}x6 + {}x7 + {}x8 + {} \
#'.format(housinggg.coef_[0], housinggg.coef_[1], housinggg.coef_[2], housinggg.coef_[3], housinggg.coef_[4], housinggg.coef_[5], housinggg.coef_[6], housinggg.coef_[7], housinggg.intercept_))


