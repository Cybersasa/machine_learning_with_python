# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 05:07:31 2024

@author: Cybersasa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=98)

#random states 1, 42, 91 yield a 100% accurate model

'''

# All this block is commented to only run the relevant part
# Before beign commented, it was run to check that every line is working as expected
x = np.array([[0, 1, 2], [2, 3, 4]])
x1 = np.array([[0, 0, 0], [2, 3, 4]])

#print("The newly created array is:\n", x)

#creating a sparse matrix (for presenting matrices with mostly 0s)

x_sparse = sparse.csr_matrix(x)
#print(x_sparse)

#Creating a sparse representation directly
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print(eye_coo)
eye_coo1 = np.array(eye_coo)
print("Eye Coo1 \n", eye_coo1)

#generating numbers between -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
#creating secondary array using sine
y = np.sin(x)
#line plot of one array against the other
plt.plot(x, y, marker = 'x')


#importing pandas which uses dataframes like in R language
#creating a simple dataset
datax = {'Name': ['John', 'Ann', 'Betty', 'ZenithX'],
         'Gender':['Male', 'Female','Female','Female'],
         'Age':[10, 11, 12, 13]}
datax_pandas = pd.DataFrame(datax)
display(datax_pandas)

#querying dataset created
print('This is utput of a query:')
display(datax_pandas[datax_pandas.Age > 12])

#import sys
#sys.path.append('machine_learning_with_python')

print('\n\n\nmglearn library imported\n')
display(datax_pandas)

#Sprint(iris_dataset)
print("Keys for Our dataset \n{}".format(iris_dataset.keys()))
#print(iris_dataset['DESCR'][:440])
print('The Classes are: {}'.format(iris_dataset['target_names']))
print('Shape of dataset is: {}'.format(iris_dataset['data'].shape))
print('First 3 rows of data:\n{}'.format(iris_dataset['data'][:3]))
print('Data Type: {}'.format(type(iris_dataset['data'])))
print('Shape of Data: {}'.format(iris_dataset['data'].shape))
print('Shape of Target: {}'.format(iris_dataset['target'].shape))
print('Shape of target Sampled: {}'.format(iris_dataset['target'][::15]))


#splitting iris dataset into 2(training and testing in ration 75:25)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)
print('Shape for training data is {} while shaep for testing data is {}.'.format(X_train.shape, X_test.shape))
print('Shape for training targets is {} while shaep for testing classes is {}.'.format(y_train.shape, y_test.shape))

#importing the K neighbours model to build a machine learning system
#this model stores the data and when presented with data point to predict, the data points in the 
#model that are closest to the input are used to give output. 
#k may be 1, 2, ... nearest points (neighbors)
from sklearn.neighbors import KNeighborsClassifier

'''
knn = KNeighborsClassifier(n_neighbors=1) #pick closest neighbour -1
knn.fit(X_train, y_train)

#new data to be predicted
Xn = np.array([[5, 2.9, 1, 0.2]])
print('New Data Shape: {}'.format(Xn.shape))

#the prediction
predicted = knn.predict(Xn)
print('Output of Prediction is {}'.format(predicted))
print('Predicted Class of {} is {}'.format(Xn, iris_dataset['target_names'][predicted]))

#checking to see if sysyem is accurate; we use the unused test data
#first check how the unused data (testing data) appears
#print(X_test, y_test)
print(X_test[0], y_test[0]) #prints the right values; first row of X_test and first row of y_test

#predict output of first test data and compare it with actual output
outA = [X_test[0]]
print(outA)
print(Xn)
print(Xn)

predicted = knn.predict(Xn)

print(type(Xn), type(X_test[0]), type(outA))

xx = knn.predict(outA)
#print(outB)
print(xx)

#actusl predicting for first value of test set
print('For first row of test data {}, the predicted value is {} while output is {}'.format(X_test[0], knn.predict([X_test[0]]), y_test[0]))
preds = []
score = []
for i in X_test:
    pred = knn.predict([i]) #knn.predict([X_test[0]])
    preds.append(pred[0])
    
for i in range(0, len(y_test)):
    if preds[i] == y_test[i]:
        score.append(1)
    print('Predicted vs Actual: {} vs {}'.format(preds[i], y_test[i]))
    
a = len(y_test)
b = len(score)
scored = (len(score)/len(y_test))*100
print('Accuracy of model is {} because out of {} predictions, {} were right.'.format(scored, a, b))
print('Accuracy1: {}'.format(knn.score(X_test, y_test)))