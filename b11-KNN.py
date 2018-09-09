#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: regkr
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1, metric = "minkowski")

"""n_neighbors ile kaç komşuya bakıp karar vereceğini, metric ile ise
mesafeyi minkowski mesafe ölçümüne göre belirledik. başka metricler için
sklearn'ün kendi tutorial'ına bakılabilir. 
Normalde komşu sayısını artırınca daha iyi sonuç alınacağı düşünülebilir.
Ama metric değeri artırılıp bunun tam tersi olduğu yani komşu sayısı artınca
başarının düştüğü confusion matrix üzerinden görülecektir.
"""
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print (cm)




    
    

