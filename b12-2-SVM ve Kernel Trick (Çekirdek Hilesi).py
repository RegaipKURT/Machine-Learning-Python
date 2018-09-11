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
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, 
                                                  random_state=0)

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

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

"""
Lineer olarak ayrılması mümkün olmayan bir veri tipi üzerinde çalışırken
SVC nin içinde belirlediğimiz kernel yani çekirdek fonksiyonunu değiştirip
farklı çekirdek kullanarak sorunu çözebiliriz. Yani aslında SVC algoritmasında
farklı çekirdek noktalar belirleyip veriyi üzüçnü bir boyuta taşıyarak
bölme işlemi gerçekleştirmeye çekirdek hilesi denir. Bunun anlaşılması için
verilerin incelenmesi gerekiyor, görselleştirme olmadan anlamak zor.
"""

from sklearn.svm import SVC
svc = SVC(kernel='poly') 
#çekirdek fonksiyonu polynomial olarak kullanmasını söyledik.
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

#bu sefer de farklı bir çekirdek fonksiyonuyla deneyelim
from sklearn.svm import SVC
svc = SVC(kernel='rbf') #poly değerini rbf olarak değiştirdik
#çekirdek fonksiyonu polynomial olarak kullanmasını söyledik.
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)








    
    

