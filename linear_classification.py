# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:19:08 2024

@author: Cybersasa
"""

#learning about linear models for classification
#starting with binary classification models for simplicity

'''

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import mglearn
from matplotlib import pyplot as plt

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10,3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,\
                                    ax=ax, alpha=0.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
axes[0].legend()

mglearn.plots.plot_linear_svc_regularization()

'''

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state=0)
#X_train1, X_test1, y_train1, y_test1 = train_test_split(cancer.data, cancer.target, random_state=0)
#cancer = LogisticRegression(max_iter=1000000, C=100).fit(X_train, y_train)
#cancer1 = LinearSVC(dual=True, max_iter=1000000, C=100).fit(X_train1, y_train1)
#print('Accuracy of Logistics Regression\n')
#print('Training: {:.2f} Testing: {:.2f}\n\n'.format(cancer.score(X_train, y_train), cancer.score(X_test, y_test)))
#print('Accuracy of Linear Support Vector Classifier (SVC)\n')
#print('Training: {:.2f} Testing: {:.2f}\n\n'.format(cancer1.score(X_train1, y_train1), cancer1.score(X_test1, y_test1)))

#Logistics regression uses L2 regularization by default
#testing Logistics regression based on L1 regularization, across range of C
#for C, marker in zip([0.001, 1, 100], ['o', 'v', '^']):
#    lr_l1 = LogisticRegression(C=C, penalty=l1).fit(X_train, y_train)
    
