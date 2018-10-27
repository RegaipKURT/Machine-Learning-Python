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

km = KMeans(n_clusters=4, init='k-means++',random_state=123)
tahmin = km.fit_predict(x)
plt.scatter(x[tahmin==0,0], x[tahmin==0,1], s=100, color="red")
plt.scatter(x[tahmin==1,0], x[tahmin==1,1], s=100, color="green")
plt.scatter(x[tahmin==2,0], x[tahmin==2,1], s=100, color="blue")
plt.scatter(x[tahmin==3,0], x[tahmin==3,1], s=100, color="black")
plt.title("KMeans")
plt.savefig("KMeans.png")
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward").fit(x)
print (ac.children_)
tahmin = ac.fit_predict(x)

plt.scatter(x[tahmin==0,0], x[tahmin==0,1], s=100, color="red")
plt.scatter(x[tahmin==1,0], x[tahmin==1,1], s=100, color="green")
plt.scatter(x[tahmin==2,0], x[tahmin==2,1], s=100, color="blue")
plt.scatter(x[tahmin==3,0], x[tahmin==3,1], s=100, color="black")
plt.title("HC Agglomerative")
plt.savefig("HC_Agglomerative.png")
plt.show()

from scipy.cluster import hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.title("Dendogram")
plt.savefig("Dendogram.png",size=1280*1024)
plt.show()
# wcss'de 2 ve 4 küme durumundaki kırılma dendogramda da görülebiliyor.
# dendogram da en uzun mesafeler yine iki ve 4 alınca oluyor. 
