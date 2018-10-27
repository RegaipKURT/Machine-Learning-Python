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
veriler = pd.read_csv("sepet.csv", header=None)

t = []
for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])

#sklearn içinde apriori algoritması için bir kütüphane olmadığından kendi indirdiğim
#apyroi kütüphanesinden import ediyorum.
from apyori import apriori
kurallar = apriori(t,min_support=0.01, min_confidence=0.02,min_lift=3, min_length=2)
#indirdiğim kod liste istediği için yukarda oluşturduğum listeye attım verileri
print (list(kurallar))
