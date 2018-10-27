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

class insan:
    def __init__(self, boy, kilo, hız):
        self.boy = boy
        self.kilo = kilo
        self.hız = hız
    def kos(self):
        self.hız = self.hız + 10

ali = insan(180, 80, 10)
ali.kos()
print ("alinin hızı:{}".format(ali.hız))
