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
import math
# 2. Veri Onisleme

# 2.1. Veri Yukleme
veriler = pd.read_csv('Ads_CTR_Optimisation.csv')
#rasgele seçim
"""
import random
satir = 10000
sutun = 10
toplamodul = 0
secilenler = []
for n in range(0,satir):
    ad = random.randrange(sutun)
    secilenler.append(ad)
    odul = veriler.values[n,ad] #verilerdeki n. satır = 1 ise ödül = 1 olacak 
    toplamodul = toplamodul + odul

print ("Kazanılan Ödül:",toplamodul)
plt.hist(secilenler)
plt.show()
"""
satir = 10000 #10 bin satır var
sutun = 10 #toplam 10 ilan var
oduller = [0] * sutun #ilk başta bütün ilanların ödülü 0
toplamoduller = 0 #toplam ödül
tiklamalar = [0] * sutun # o ana kadarki tıklamalar
secilenler = []

for n in range(0,satir):
    ad = 0 #seçilen ilan
    max_ucb = 0
    for i in range(0,sutun): 
        if(tiklamalar[i]>0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2 * (math.log(n)/tiklamalar[i]))
            ucb = ortalama + delta
        else:
            ucb = satir * 10
        if max_ucb < ucb: #max_ucb'den büyük ucb varsa
            max_ucb =ucb #artık max_ucb ona eşit olsun
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n,ad]
    tiklamalar[ad] = tiklamalar[ad] + 1
    oduller[ad] = oduller[ad] + odul
    toplamoduller = toplamoduller + odul

print ("Toplam Ödül:",toplamoduller)

plt.hist(secilenler)
plt.title("Seçimler")
plt.show()