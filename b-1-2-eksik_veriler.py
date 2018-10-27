# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 18:43:36 2018

@author: regkr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer #eksik verileri tamamlamak için
veriler = pd.read_csv("eksikveriler.csv")  #dosya adımızı yazıp değişkene atadık
kilo = veriler[["kilo"]] #verilerin içinde kilo kısmını aldık
print (veriler)
#print (kilo)
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
Yas = veriler.iloc[:,1:4].values
print (Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas = imputer.transform(Yas[:,1:4])
print (Yas)
