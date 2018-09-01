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
veriler = pd.read_csv('maaslar.csv')
#pd.read_csv("veriler.csv")
print (veriler)
#veri on isleme
y = veriler.iloc[:,-1:].values
print (y)
x = veriler.iloc[:,-2:-1].values
print (x)
""" #burayı kendi grafiğimi çizmek için yapmıştım.
plt.plot(x,y)
plt.xlabel("Eğitim Seviyesi")
plt.ylabel("Maaş")
plt.title("Eğitim Seviyesi - Maaş İlişkisi")
plt.savefig("Egitim_Maas.png")
plt.show()
"""
#lineer regresyon ile yapsak nasıl sonuç çıkarmış onu oluşturup çizdik.
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg.predict(x),color="blue")
plt.title("Doğrusal Regresyon Tahmini")
plt.show()

#polinomal regresyon burada başlıyor.
from sklearn.preprocessing import PolynomialFeatures
poli_reg = PolynomialFeatures(degree=2) #2. dereceden polinom oluşturmak.
x_poli = poli_reg.fit_transform(x)
print (x_poli)
lin_reg2 = LinearRegression() #lineer regresyon objesi oluşturmak.
lin_reg2.fit(x_poli,y) #lineer regresyon objesini eğitmek. polinomal olduğu için x_poli yapmıştık
#görselleştirme aşaması
plt.scatter(x,y,color = "red")
plt.plot(x, lin_reg2.predict(poli_reg.fit_transform(x)), color ="blue")
#üstte hem scatter hem plot grafiği çizdik, ikisi de aynı grafikte üst üste görünecek.
plt.title("2. Derece Polinomal Regresyon Tahmini")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poli_reg = PolynomialFeatures(degree=4) #4. dereceden polinom oluşturmak.
x_poli = poli_reg.fit_transform(x) #x değerlerini polinomal hale çeiviryoruz.
print (x_poli) #polinomal regresyon kullanacağımız için polinomal veri oluşturmamız lazım çünkü.

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poli,y) #polinomal haliyle eğittiğimize dikkat et!!!

plt.scatter(x,y,color = "red")
plt.plot(x, lin_reg2.predict(poli_reg.fit_transform(x)), color ="blue")
plt.title("4. Derece Polinomal Regresyon Tahmini")
plt.show()

#tahminler yaparsak nasıl sonuçlar verirler
#üzerinde düzgün çıkması için oynadım biraz. Şablonda orjinal hali var.
#numpy dizisini int değere çevirdim düzgün görünsün diye.
print ("\nDoğrusal regresyona göre verilecek maaşlar:")
print("11.seviye için:",int(lin_reg.predict(11)[0])) #11.seviyeye ne maaş verir
print("6.6 seviyesi için:",int(lin_reg.predict(6.6)[0])) #6.6 seviyesine ne maaş verilir tahimin et

print ("\nPolinomal regresyona göre verilecek maaşlar:")
print ("11.seviye için:",int(lin_reg2.predict(poli_reg.fit_transform(11))[0]))
print ("6.6 seviyesi için:",int(lin_reg2.predict(poli_reg.fit_transform(6.6))[0]))

#burada linear regression için kullandığımız metodun aynısını kullanıyoruz ama
#sadece tahmine vermeden önce bir dönüşüm yapıyoruz.
