#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:03:40 2018

@author: regkr
"""
"""
PCA bir bir boyut indirgeme yöntemidir. Bütün makine öğrenmesi kütüphanelerinin
içinde PCA algoritması bulunabilir. Burada yapılan işlem verinin içindeki kolonları
elemekten başka bir şey değildir aslında. Gerekesiz olan kolonlar bu algoritmanın
kendi mantığıyla elenir. Ama bu aynı zamanda veri kaybına da neden olabilir.
"""
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('Wine.csv')
#pd.read_csv("veriler.csv")

#veri on isleme
X= veriler.iloc[:,0:13].values
Y = veriler.iloc[:,13].values

#encoder:  Kategorik -> Numeric

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train2 = pca.fit_transform(X_train)
#test verisine de boyut indirgeme uygulamamız lazım ki 
#test sonuçlarıyla aynı boyutlu uzayda uyuşsunlar
X_test2 = pca.transform(X_test) 
#bu sefer eğitmeden transfor ettik 

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0,)
lr.fit(X_train,y_train)
#PCA uygulanmamış veri
lr2 = LogisticRegression(random_state=0)
lr2.fit(X_train2,y_train)

#TAHMİNLER
y_pred = lr.predict(X_test)
y_pred2 = lr2.predict(X_test2)


from sklearn.metrics import confusion_matrix

print ("Orjinal Tahmin Matirisi:")
cm1 = confusion_matrix(y_test, y_pred)
print (cm1)
print ("PCA Uygulanmış Tahmin:")
cm2 = confusion_matrix(y_test, y_pred2)
print (cm2)

"""
Sonuçta görülüyor ki 13 kolondan iki kolona indirilince tek fark
sadece 1 tane fazladan hata yapılıyor. Yani veri 6'da 1'ine iniyor
ve buna rağmen sadece 1 hata artıyor. Burada hata artışı olduğu doğru
ama başka veriler de hatayı azaltabilir unutmamak lazım ki.
"""


#LDA
"""
LDA ile PCA ile arasındaki en büyük fark PCA için bir sınıf farkı yoktur.
Yani PCA bir gözetimsiz öğrenmedir. LDA için ise sınıflar birbirinden 
farklıdır. LDA sınıfları en iyi ifade eden algoritmayı bulur. PCA ise verinin
tamamını en iyi ifade eden algoritmayı bulur.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
lda = LinearDiscriminantAnalysis(n_components=2) #iki boyuta indirdik
X_train3 = lda.fit_transform(X_train, y_train)
X_test3 = lda.fit_transform(X_test, y_test)
#lda y verisini de alıyor DİKKAT ET! OYSA PCA Y almaz.
#LDA Uygulanmış Logistic REgression tahminleri
lr3 = LogisticRegression(random_state=0)
lr3.fit(X_train3, y_train)
y_pred3 = lr3.predict(X_test3)

#LDA Uygulanmamış ORJİNAL tahminler zaten yukarda var.

print ("LDA Uygulanmış Tahmin:")
cm3 = confusion_matrix(y_test, y_pred3)
print (cm3)

"""
LDA Sınıf fdarkını da gözeterek boyut indirgeme yaptığı için orjinal
verideki tahiminlerin aynısını %100 başarı ile yaptı.
"""














