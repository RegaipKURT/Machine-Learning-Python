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
from sklearn.metrics import r2_score #r kare değerini hesaplamak için

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


print("Linear R2 degeri:")
print(r2_score(Y, lin_reg.predict((X))))

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


print("Polynomial R2 degeri:")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X)) ))


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

print("SVR R2 degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )


#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4

plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')
plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K), color = 'black')
plt.show()
print(r_dt.predict(11))
print(r_dt.predict(6.6))

print("Decision Tree R2 degeri:")
print(r2_score(Y, r_dt.predict(X)) )



#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(X,Y)

print(rf_reg.predict(6.6))

plt.scatter(X,Y, color='red')
plt.plot(x,rf_reg.predict(X), color = 'blue')
plt.plot(x,rf_reg.predict(Z), color = 'green')
plt.plot(x,r_dt.predict(K), color = 'black')
plt.show()
#plt.show dersek kodun geri kalanını grafiği çizdikten sonra çalıştırır
#aksi takdirde grafik geç çizildiği için en sonda görünür
#ayrıca plt.show demeden grafik çizdirmeye devam edersek grafikler üst üste çizilir.

print('--------------------------')
print("Random Forest R2 degeri:")
print(r2_score(Y, rf_reg.predict(X)) )
print(r2_score(Y, rf_reg.predict(K)) )
print(r2_score(Y, rf_reg.predict(Z)) )

#özet R2 degerleri
print('--------------------------')
print("Linear R2 degeri:")
print(r2_score(Y, lin_reg.predict((X))))
#r2_score gerçek değeri ve tahmin değerini alır fonksiyon olarak.

print('---------------------------')
print("Polynomial R2 degeri:")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X)) ))

print('---------------------------')
print("SVR R2 degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )

print('---------------------------')
print("Decision Tree R2 degeri:")
print(r2_score(Y, r_dt.predict(X)) )
print('---------------------------')
print("Random Forest R2 degeri:")
print(r2_score(Y, rf_reg.predict(X)) )

from sklearn.metrics import r2_score
r2_score(Y, rf_reg.predict(X))















