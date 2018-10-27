#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: regkr
"""

#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme

veriler = pd.read_csv('eksikveriler.csv')
#pd.read_csv("veriler.csv")

print(veriler)

#veri on isleme
boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

x = 10

class insan:
    boy = 180
    def kosmak(self,b):
        return b + 10

ali = insan()
print(ali.boy)
print(ali.kosmak(90))


#eksik veriler
#sci - kit learn
from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )    

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

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
print(list(range(22)))

sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet , index=range(22), columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)


#veri bölmek için gerekli kütüphane(farklı bölme yöntemleri de var)
from sklearn.cross_validation import train_test_split
#ülke, boy, kilo ve yaş bilgilerini kullanrak cinsiyeti bulacaz.
#dolayısıyla bağımlı ve bağımsız değişkenler var.
#bu yüzden bulmak istediğimiz kolonla elimizdeki verileri ayrı ayrı ele alacaz.
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)
#x_train, x_test ve y_train,_y test bizim verdiğimiz isimler
#train_tast_split içine sonuc3 ve s i vererek böleceğiz.
#s bizim hazırlayıp son hale getirdiğimiz veri kümesi sonuc3 ise bulmak istediğimiz kolon
#test_size=0.33 verinin %33'ü test için kullanılacak demek
#random state =0 ise bir bölme yöntemi ve rastgele her durumdan örnekler seç demektir.
#random state kullanmazsak fransa alt taraflarda olduğu için fransadan hiç örnek almayabilirdi.
#y bağımlı x bağımsız değişken olarak tanımlandı(değiştirilebilir ismi, sıkıntıu değil.)

from sklearn.preprocessing import StandardScaler
#veriler farklı kümelere ait olduğu için standardize etmemiz lazım.
#standardizasyon ve normalizasyon yöntemlerinden standardizasyon yöntemini
#kullandık, çünkü uç veriler olduğunda daha iyi sonuç veriyor.
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


