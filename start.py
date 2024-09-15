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

'''
from sklearn.datasets import load_iris
