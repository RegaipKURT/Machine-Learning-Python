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
from sklearn.metrics import confusion_matrix

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_excel('Iris.xls')
#pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, 
                                                  random_state=0)
y_train = y_train.ravel() 
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Buradan itibaren sınıflandırma algoritmaları başlar
# 1. Logistic Regression

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train) #egitim

y_pred = logr.predict(x_test) #tahmin

#karmasiklik matrisi
cmlin = confusion_matrix(y_test,y_pred)
print("SONUÇLAR:\n-------------------------------")
print("Lineer Model Karmaşıklık Matrisi:")
print(cmlin,"\n-------------------------------")


# 2. KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cm_knn = confusion_matrix(y_test,y_pred)
print("KNN Karmaşıklık Matrisi:")
print(cm_knn,"\n-------------------------------")

# 3. SVC (SVM classifier)
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

cm_svc = confusion_matrix(y_test,y_pred)
print('SVC Karmaşıklık Matrisi:')
print(cm_svc,"\n-------------------------------")

# 4. NAive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

cm_gnb = confusion_matrix(y_test,y_pred)
print('GNB Karmaşıklık Matrisi:')
print(cm_gnb,"\n-------------------------------")

# 5. Decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'gini')

dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm_dt = confusion_matrix(y_test,y_pred)
print('DTC Karmaşıklık Matrisi:')
print(cm_dt,"\n-------------------------------")

# 6. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=11, criterion = 'gini')
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
cm_rfc = confusion_matrix(y_test,y_pred)
print('RFC Karmaşıklık Matrisi:')
print(cm_rfc,"\n-------------------------------")

# 7. ROC , TPR, FPR değerleri 
#proba tahmin olsaılıkları matrisidir. yüzde kaç erkektir örneğin. (ex: 0.2)
y_proba = rfc.predict_proba(X_test)

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],
                                      pos_label='Iris-setosa')
print("False Positive Rate:\n",fpr,"\n-------------------------------")
print("True Positive Rate:\n",tpr,"\n-------------------------------")
print("Treshold:\n",thold,"\n-------------------------------")




