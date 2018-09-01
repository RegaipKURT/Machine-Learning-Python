# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:28:23 2018

@author: regkr
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

satislar = pd.read_csv("satislar.csv")

print (satislar)

ay = satislar[["Aylar"]]
satis = satislar[["Satislar"]]

print (ay, "\n",satis)

"""
satis2 = satislar.iloc[:,1:].values
ay2 = satislar.iloc[:,:1].values

print (satis2)
print (ay2)
"""

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(ay,satis,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

#eğitim ve test için kullanacağımız verileri standardize ediyoruz
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
Y_train = sc.fit_transform(y_train)
X_test = sc.fit_transform(x_test)
Y_test = sc.fit_transform(y_test)
#sklearn içinden basit doğrusal regresyon için öğrenme kütüphanesini ekliyoruz
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)








