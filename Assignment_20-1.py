#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:05:46 2018

@author: macuser
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

with open("nba_2013.csv", 'r') as csvfile:
    nba_data = pd.read_csv(csvfile)

# Design Matrix X - Features - g, gs, mp, fg, fga
X = np.array(nba_data.iloc[:,4:9])
# Target Vector Y - pts
Y = np.array(nba_data.iloc[:,28:29])

X_train, X_test, y_train, y_test = train_test_split(
X, Y, test_size = 0.3, random_state = 100)
y_train = y_train.ravel()
y_test = y_test.ravel()

for K in range(25):
    K_value = K+1
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform')
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)
