# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 05:07:31 2024

@author: Sana
"""

import numpy as np

x = np.array([[0, 1, 2], [2, 3, 4]])
x1 = np.array([[0, 0, 0], [2, 3, 4]])

#print("The newly created array is:\n", x)

#creating a sparse matrix (for presenting matrices with mostly 0s)

from scipy import sparse
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

#sample of how matplotlibworks for visualizations
#%matplotlib inline
import matplotlib.pyplot as plt
#generating numbers between -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
#creating secondary array using sine
y = np.sin(x)
#line plot of one array against the other
plt.plot(x, y, marker = 'x')