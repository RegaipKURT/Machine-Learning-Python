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
veriler = pd.read_csv('eksikveriler.csv')
#pd.read_csv("veriler.csv")

#veri on isleme

#eksik veriler

#encoder:  Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

c = veriler.iloc[:,-1:].values
print(c)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
print(c)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)
#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1] , index=range(22), columns=['cinsiyet'])
print(sonuc3)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
print (boy)

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)




    
    

