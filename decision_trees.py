# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:44:45 2024

@author: Cybersasa
"""

#import mglearn
#import sklearn
#mglearn.plots.plot_animal_tree()
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#splitting breast cancer dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(load_breast_cancer().data, load_breast_cancer().target, stratify=load_breast_cancer().target, random_state=42)
cancer_tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print('Accuracy of Decision tree is {:.2f}'.format(100*cancer_tree.score(X_test, y_test)))
#getting maximum value the model can create for test set
#accuracies = []
#for i in range (1,100):
 #   X_train, X_test, y_train, y_test = train_test_split(load_breast_cancer().data, load_breast_cancer().target, stratify=load_breast_cancer().target, random_state=i)
  #  cancer_tree = DecisionTreeClassifier(random_state=1).fit(X_train, y_train)

   # accuracy = cancer_tree.score(X_test, y_test)
    #accuracies.append(accuracy)
#print('Max Accuracy is {:.2f}'.format(100*max(accuracies)))







