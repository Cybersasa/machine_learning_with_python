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
print('Accuracy of Decision tree (training) is {:.2f}'.format(100*cancer_tree.score(X_train, y_train)))
#getting maximum value the model can create for test set
#accuracies = []
#for i in range (1,100):
 #   X_train, X_test, y_train, y_test = train_test_split(load_breast_cancer().data, load_breast_cancer().target, stratify=load_breast_cancer().target, random_state=i)
  #  cancer_tree = DecisionTreeClassifier(random_state=1).fit(X_train, y_train)

   # accuracy = cancer_tree.score(X_test, y_test)
    #accuracies.append(accuracy)
#print('Max Accuracy is {:.2f}'.format(100*max(accuracies)))

#pre-pruning the decision tree by adding a maximum depth

X_train, X_test, y_train, y_test = train_test_split(load_breast_cancer().data, load_breast_cancer().target, stratify=load_breast_cancer().target, random_state=42)
cancer_tree = DecisionTreeClassifier(random_state=0, max_depth=7).fit(X_train, y_train)
print('Accuracy of Decision tree is {:.2f}'.format(100*cancer_tree.score(X_test, y_test)))
print('Accuracy of Decision tree (training) is {:.2f}'.format(100*cancer_tree.score(X_train, y_train)))

#after adding max depth of 4, the accuracy of predictions increases 93.71% to 95.10%
from sklearn.tree import export_graphviz
export_graphviz(cancer_tree, out_file="tree.dot", class_names=["malignant", "benign"], feature_names=load_breast_cancer().feature_names, impurity=False, filled=True)

print('Features: {}'.format(load_breast_cancer().feature_names[:4]))
import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)




