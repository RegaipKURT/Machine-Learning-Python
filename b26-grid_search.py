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

# veri kümesi
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM
print ("\nSVC Çalışıyor...")
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Tahminler
y_pred = classifier.predict(X_test)

#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("SVC Confusion Matrisi:\n",cm)

#k-katlamali capraz dogrulama 
from sklearn.model_selection import cross_val_score
''' 
1. estimator : classifier (bizim durum) çünkü svc objesine classifier ismini verdik.
2. X
3. Y
4. cv : kaç katlamalı
'''
print ("\nK-FOLD çalışıyor...")
basari = cross_val_score(estimator = classifier, X=X_train, y=y_train , cv = 4)
print ("Başarıların ortalması:")
print(basari.mean()) #mean yüntemi başarıların ortalamasını verir.
print ("Başarıların Standart Sapması:")
print(basari.std()) # std ise başarılar arasındaki standart sapmayı verir. (düşük olmalı)

#Grid Search (ızgara araması) ile Parametre optimizasyonu 
print ("\nGridSearch (Izgara Araması) Çalışıyor...")
from sklearn.model_selection import GridSearchCV
#grid serch içine aramasını istediğimiz parametreleri bir liste olarak veriyoruz.
p = [{"C":[1,2,3,4,5], "kernel":["linear","poly","sigmoid","rbf"]},
      {"C":[1,10,100,1000], "kernel":["rbf"], "gamma":[1,0.5,0.1,0.001]}]
#önce liste, sonra sözlük, sonra liste şeklinde DİKKAT ET!
"""
GS parametreleri
estimator: neyi optimize etmek istiyoruz
param_grid: parametreler / denenecekler bizim listemiz
scoring: neye göre skorlanacak. ör:accuracy
n_jobs: aynı anda kaç iş çalışacak?
"""
gs = GridSearchCV(estimator=classifier, #svm'ye verdiğimiz isim
                  param_grid= p,scoring= "accuracy")

#aslında svm işin özünde çalışacak ama gridsearch bunun üstünde çalışarak optimize etmeye çalışacak
grid_search = gs.fit(X_train,y_train)
eniyisonuc = grid_search.best_score_
eniyiparametreler = grid_search.best_params_
print ("En iyi sonuç:\n",eniyisonuc)
print ("En iyi parametreler:\n",eniyiparametreler)


