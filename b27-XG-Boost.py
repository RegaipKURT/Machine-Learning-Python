#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:03:40 2018

@author: regkr
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
#pd.read_csv("veriler.csv")

#veri on isleme
X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

#encoder:  Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X = X[:,1:]


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)


#XG-BOOST 
from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(x_train,y_train)

y_pred = xg.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print (cm)











