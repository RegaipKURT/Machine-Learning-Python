#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: regkr
"""
# 1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# 2.1. Veri Yukleme
veriler = pd.read_csv('Restaurant_Reviews.csv')
ps = PorterStemmer()
derlem = []

for i in range(1000):
    yorum = re.sub("[^a-zA-Z]", " ", veriler["Review"][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum = " ".join(yorum)
    derlem.append(yorum)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray()
Y = veriler.iloc[:,1:]

from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X, Y,test_size=0.2, random_state=0)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('\nGNB')
print (cm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('\nKNN')
print(cm)

from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('\nSVC')
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('\nDTC')
print(cm)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=11, criterion = 'entropy')
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('\nRFC')
print(cm)
