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

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#tahminler

print(lin_reg.predict(11))
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()
print(svr_reg.predict(11))
print(svr_reg.predict(6.6))


from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.5
plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')
plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K), color = 'black')
plt.show()
print(r_dt.predict(11))
print(r_dt.predict(6.6))

from sklearn.ensemble import RandomForestRegressor #rf ,mport işlemi
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
"""
RandomForestRegressor(n_estimators = 10, random_state=0) anlamı:
içine verdiğimiz değerlerden n_estimator kaç tahmin yapılacağı veya kaç
tane decision tree çizileceği, random state ise verileri random olarak 
böleceğini anlatıyor.
"""
rf_reg.fit(X,Y) #modeli x ve y den eğit

print(rf_reg.predict(6.6)) #rf ile 6.6 ya gelen tahmin değerini yazdır

#grafikleri çizdirme
plt.scatter(X,Y, color='red') #scatter data pointleri yani veri noktalarını çizer.
#plot ise grafik çizer noktalar yerine
plt.plot(x,rf_reg.predict(X), color = 'blue')
plt.plot(x,rf_reg.predict(Z), color = 'green')
plt.plot(x,rf_reg.predict(K), color = 'black')
#görüleceği gibi farklı ağaçlardan farklı sonuçlar çıktı
#random forest farklı ağaçlardan gelen değerlerin ortalamasını verir.
"""
Bildiğimiz verilerde dt algoritması daha iyi gibi görünebilir
hatta karşılaştırma yaptığımızda randomforest daha başarısız görünebilir
ama bu sadece burası için geçerlidir. çünkü eğer bilmediğimiz verilerden
tahmin yaparsak decision tree bildiği verilerle aynı sonuçları döndürme
eğilimine sahiptir. Ama randomforest birçok decision tree nin ortalamasını
döndürdüğü için daha iyi sonuçlar verecektir.
"""






    

