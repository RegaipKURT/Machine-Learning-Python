# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 18:43:36 2018

@author: regkr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer #eksik verileri tamamlamak için
veriler = pd.read_csv("eksikveriler.csv")  #dosya adımızı yazıp değişkene atadık
kilo = veriler[["kilo"]] #verilerin içinde kilo kısmını aldık
print (veriler)
#print (kilo)
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
Yas = veriler.iloc[:,1:4].values #1 ile 4. kolonlar arasını aldık. Çünkü numerik olanlar sadece onlar.
print (Yas)
imputer = imputer.fit(Yas[:,1:4]) #1 ile 4. kolonlar arasına imputer fonksiyonunu uygula dedik.
Yas = imputer.transform(Yas[:,1:4]) #şimdi de uyguladığımız haline çevirdik veriyi.
print (Yas)

ulke = veriler.iloc[:,0:1].values #ulke değişkenine ilgili kolondaki değerleri atadık.
print (ulke)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder() #LabelEncoder verilen değerleri sayıya çevirir.
ulke [:,0] = le.fit_transform(ulke[:,0])
print (ulke)

ohe = OneHotEncoder(categorical_features="all") #Bu encoder kolon bazlı sayılara çevirir.
ulke=ohe.fit_transform(ulke).toarray()
print (ulke)

#Buradan sonrası veri birleştirmenin nasıl olacağı gösteriliyor.
#ülkelere göre ayırdıktan sonra kolonlara başlık verip sonuç içine atadım 
sonuc = pd.DataFrame(data = ulke, index=range(22), columns = ["fr","tr","us"])
print (sonuc)
#Yas değişkeninin de kolonlarına isim verdik burada
sonuc2 = pd.DataFrame(data = Yas, index=range(22), columns = ["kilo","yas"])
print (sonuc2)
#cinsiyet verisini almamıştım şimdi alıyorum
cinsiyet = veriler.iloc[:,-1:].values
print (cinsiyet)
#cinsiyet kolonuna bir isim verdim
sonuc3 = pd.DataFrame(data = cinsiyet, index=range(22), columns =["cinsiyet"])
print (sonuc3)
#şimdi s değişkeninde sonuc ve sonuc2 yi birleştiriyoruz pandas.concat ile
s=pd.concat([sonuc,sonuc2],axis=1) #axis ile satıları düzenliyoruz.
print (s)
#cinsiyeti sonuc3'e atamıştım onu da s ile tekrar birleştiriyorum.
s2 = pd.concat([s,sonuc3],axis=1)

print (s2)







