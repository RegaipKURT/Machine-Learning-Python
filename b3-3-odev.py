#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")
print (veriler)
#veri on isleme

#eksik veriler

#encoder:  Kategorik -> Numeric
#bütün kolonlara aynı anda label encodinguygulama
veriler2 = veriler.apply(LabelEncoder().fit_transform)

c = veriler2.iloc[:,0:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)

havadurumu = pd.DataFrame(data = c, index = range(14), columns = ["O","R","S"])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]], axis=1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)

from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

import statsmodels.formula.api as sm
#veriler içine 1 lerden oluşan bir kolon ekliyoruz.(sabit oluşturmak için)
X = np.append(arr = np.ones((14,1)).astype(int), values = sonveriler.iloc[:,:-1], axis = 1)

X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
#X_l'deki kolonların boy üzerindeki etkisini ölçmek için kullandığımız kod
r_ols = sm.OLS(endog=sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print (r.summary())
sonveriler = sonveriler.iloc[:,1:]
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
#X_l'deki kolonların boy üzerindeki etkisini ölçmek için kullandığımız kod
r_ols = sm.OLS(endog=sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print (r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
"""
#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = hava, index = range(14), columns=['günes','bulut','yagmur'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = oyun, index = range(14), columns = ['var','yok'])
print(sonuc2)

nem = veriler.iloc[:,2:3].values
print(nem)

sicaklik = veriler.iloc[:,1:2].values
print(sicaklik)

rüzgar = veriler.iloc[:,-2:-1].values
print (rüzgar)

sonuc3 = pd.DataFrame(data = nem[:,:1] , index=range(14), columns=['nem'])
print(sonuc3)

sonuc4 = pd.DataFrame(data = sicaklik[:,:1] , index=range(14), columns=['sicaklik'])
print(sonuc4)

sonuc5 = pd.DataFrame(data = rüzgar[:,:1] , index=range(14), columns=['rüzgar'])
print(sonuc4)
#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)
s3= pd.concat([s2,sonuc4],axis=1)
print(s3)
s4 = pd.concat([s3,sonuc5],axis=1)
print (s4)

oyun = s4.iloc[:,3:4].values
print (oyun)

sol = s4.iloc[:,:3]
print (sol)
sag = s4.iloc[:,5:]
print (sag)

veri = pd.concat([sol,sag],axis=1)
print (veri)
#verilerin egitim ve test icin bolunmesi

from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(veri,oyun,test_size=0.33, random_state=0)

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


#backward elimination yöntemi
import statsmodels.formula.api as sm
#veriler içine 1 lerden oluşan bir kolon ekliyoruz.(sabit oluşturmak için)
X = np.append(arr = np.ones((14,1)).astype(int), values = veri, axis = 1)

X_l = veri.iloc[:,[0,1,2,3,4]].values
#X_l'deki kolonların boy üzerindeki etkisini ölçmek için kullandığımız kod
r_ols = sm.OLS(endog=oyun, exog =X_l)
r = r_ols.fit()
print (r.summary())


#şimdi burada yukarıdaki kodu çalıştırınca p değerinin en yüksek olduğu 4. kolunu
#listeden çıkardık ve kalan kolonlarla tekrar etkileri ölçtük.
X_l = veri.iloc[:,[0,1,2,3,]].values
r_ols = sm.OLS(endog=boy, exog =X_l)
r = r_ols.fit()
print (r.summary())

#tekrar ölçtüğümüzde son kolonunda p değerinin 0'dan büyük olduğunu gördük ve
#elemek istedik. Aslında 0.05 in altındaydı yani kabul edilebilridi ama eledik yine de
X_l = veri.iloc[:,[0,1,2,3,5]].values
r_ols = sm.OLS(endog=boy, exog =X_l)
r = r_ols.fit()
print (r.summary()) #rapor veya özet yazdırdık modelle ilgili
"""

    
