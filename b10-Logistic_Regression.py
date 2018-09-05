# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#verilerin yüklenmesi
veriler = pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values #x'i boy, kilo ve yaş kolonlarından oluşturduk
y = veriler.iloc[:,-1].values #y değerini ise cinsiyet kolonu olarak aldık

#verilerin eğitim-test olarak ayrılması
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=0)

#verilerin ölçeklenmesi (belki gerek olmayabilir.)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(x_train) 
#eğiteceğimiz veri olduğundan fit_transform kullandık
X_test = sc.transform(x_test)
#test olduğu için direk transform yaptık.

#modelin oluşturulması
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) #modelin eğitilmesi bölünmüş verilerden

y_pred = logr.predict(X_test) #x_testten tahmin ettiği sonuçları y_pred'e atadık
print (y_pred)
print (y_test)
#buraya kadar olsan kısımda ölçeklenmemiş değerler daha doğru sonuç verdi.
# ama şu an ölçekli hali yüklü.

#buradan sonrasında ise confusion matrix ile başarı ölçeceğiz
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print (cm)
"""
cm nin asal köşegenleri toplamı doğru tahmin sayısını verirken,
kalan köşegenlerindeki sayıların toplamı ise yanlış tahmin sayısını verir.
yani algoritma başarısı = asal toplamlar / matrisin tümü denilebilir.
şekille şöyle gösterebiliriz:
[doğru      yanlış]
[yanlış      doğru]
"""


