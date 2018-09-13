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
# pd.read_csv("veriler.csv")
x = veriler.iloc[:,2:].values

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward").fit(x)
print (ac.children_)
tahmin = ac.fit_predict(x)

plt.scatter(x[tahmin==0,0], x[tahmin==0,1], s=100, color="red")
plt.scatter(x[tahmin==1,0], x[tahmin==1,1], s=100, color="green")
plt.scatter(x[tahmin==2,0], x[tahmin==2,1], s=100, color="blue")
plt.show()