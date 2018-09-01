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

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

#data frame dilimleme (slice) verinin içinden bir kolonu alma
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NumPY dizi (array) dönüşümü
X = x.values
Y = y.values


#linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y) #objemizi x ve y değerlerinden eğitiyoruz.

#polynomial regression
#doğrusal olmayan (nonlinear model) oluşturma
#2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# 4. dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

# Gorsellestirme: model inşalarının sırasına göre görselleştirdik.
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()

#tahminler
#doğrusal regresyona göre
print(lin_reg.predict(11)) #11. seviyeye verilecek maaşı tahmin et
print(lin_reg.predict(6.6)) #6.6 seviyesine verilecek maaşı tahmin et

#polinomal regresyona göre
print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))









    

