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
