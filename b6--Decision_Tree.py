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

r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y,color = "red")
plt.plot(X,r_dt.predict(X),color = "blue")
plt.plot(X,r_dt.predict(Z),color = "green") #üç çizgide aynı noktadan geçiyor
plt.plot(X,r_dt.predict(K),color = "black") #üç renk üst üste çıkacak
print(r_dt.predict(11))
print(r_dt.predict(6.6))
#3 renk de aynı grafikte üst üste bunun sebebi değerleri hep aynı aralığa indirdi.
#KARAR AĞACI BÖLEGELRE BÖLÜYOR VERİYİ, GRAFİKLERİN ÜST ÜSTE ÇIKMASININ SEBEBİ AYNI BÖLGEYE DENK GELMELERİ
#K'YI X - 1 YAPARSAK FARKLI BÖLGEYE DENK GELDİĞİNİ GÖREBİLİRİZ.
"""
K = X - 1
Z = X + 0.5
plt.scatter(X,Y,color = "red")
plt.plot(X,r_dt.predict(X),color = "blue")
plt.plot(X,r_dt.predict(Z),color = "green")
plt.plot(X,r_dt.predict(K),color = "black")
print(r_dt.predict(11))
print(r_dt.predict(6.6))
"""

    

