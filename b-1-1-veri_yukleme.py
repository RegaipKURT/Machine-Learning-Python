# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 18:43:36 2018

@author: regkr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv")  #dosya adımızı yazıp değişkene atadık
kilo = veriler[["kilo"]] #verilerin içinde kilo kısmını aldık
print (veriler)
print (kilo)

boy = veriler[["boy"]] #niye iki parantez??? Çünkü tek parantezde yazınca 
print (boy)            #verinin üstüne boy diye yazmıyor.Yoksa tek de olabilir.

