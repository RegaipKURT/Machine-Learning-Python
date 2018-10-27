#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: regkr
"""
# 1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Veri Onisleme

# 2.1. Veri Yukleme
veriler = pd.read_csv('musteriler.csv')
x = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, init='k-means++').fit(x)
print (km.cluster_centers_) #küme merkez noktalarnı yazdır.

"""
kmeans ile işimiz burada bitti. fakat "WCSS" değerini belirlemek için bir 
grafik çizerek sonuçları görebilir ve oluşturulacak küme sayısını ona göre
optimize edebiliriz.
WCSS: Within Cluster Some of Squares yani;
her bir datapoint'in kümenin ağırlık merkezine olan uzaklıklarının karesi.
"""

sonuclar = [] #bir listeye wcss değerlerini atayıp grafik çizdireceğiz.
for i in range(1,10): #1'den 10'a kadar küme sayısıyla oluşan wcss değerlerini listeye atacağız.
    km = KMeans(n_clusters=i, init='k-means++',random_state=123)
    km.fit(x)
    sonuclar.append(km.inertia_) #.inertia_ wcss değerini bulur

print (sonuclar)

#grafiğini çizdirelim
plt.plot(range(1,10),sonuclar)
plt.ylabel("WCSS DEĞERİ")
plt.xlabel("KÜME SAYISI")
plt.title("WCSS GRAFİĞİ")
plt.savefig("wcss_grafik.png")
plt.show() #bu grafik üzerindeki dirsek noktalarını küme sayısı olarak alabiliriz.


