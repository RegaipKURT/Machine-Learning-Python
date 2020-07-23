# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:34:11 2020

@author: regkr
"""

# KÜTÜPHANELER
from keras.datasets import mnist #EL YAZISI RAKAMLARIN BULUNDUĞU VERİSETİ
from keras.layers import Dense # BASİT YAPAY SİNİR AĞI KATMANI
from keras.models import Sequential # SIRALI YSA MODELİ
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import SGD, Adam #OPTİMİZASYON FONKSİYONLARI
import seaborn as sns #GÖRSELLEŞTİRME KÜTÜPHANESİ
sns.set_style("darkgrid") #KOYU GİRİ IZGARALI ARKAPLAN
import warnings
warnings.filterwarnings("ignore") 

(x_train, y_train), (x_test, y_test) = mnist.load_data() #VERİSETİNİN YÜKLENMESİ

x_train = x_train.reshape(60000, 28*28) #TRAİN VERİSETİNİ MATRİSTEN VEKTÖRE ÇEVİRDİK.
x_test = x_test.reshape(10000, 28*28) #TEST VERİSETİNİ MATRİSTEN VEKTÖRE ÇEVİRDİK.

x_train = x_train / 255.0 #NORMALİZASYON UYGULADIK
x_test = x_test / 255.0
"""
Bir renkli görüntü 255 tane sayıyla temsil edilen değerlerden oluşur. Bu değerleri 0 ile 1 arasına çekmek için
255'e böldük. Verilere uygulanan bu ölçeklendirme işlemi yakınsama hesaplamalarının daha hızlı yapılmasını sağlar.
"""

model = Sequential() #MODELİ TANIMLADIK

model.add(Dense(128, activation="relu", input_shape=(28*28,))) #KATMAN EKLEDİK VE 128 NÖRONDAN OLUŞUYOR.
model.add(Dense(10, activation="softmax")) #ÇIKIŞ KATMANI 10 TANE NÖRONDAN OLUŞUYOR. ÇÜNKÜ 10 TANE DEĞER TAHMİN EDECEĞİZ.

model.compile(optimizer=Adam(learning_rate=0.0005), loss="sparse_categorical_crossentropy", metrics=["acc"])
#MODELİ DERLERKEN ADAPTİVE MOMENTUM OPTİMİZASYONU VE KULANACAĞIMIZ LOSS FONKSİYONUNU BELİRLEDİK.
"""
Bir YSA modeli kurulurken kullanılacak olan loss fonksiyonu modelin yapısına uygun olmalıdır. Aksi halde model ya çalışmaz
ya da hatalı sonuçlar üretir. Eğitim başlamadan alacağınız boyut hataları çoğu zaman loss fonksiyonlarından kaynaklanır.
"""

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test)) #MODEL EĞİTİMİ
"""
Daha sonra acccuracy ve loss değerlerini alabilmek için modeli history isimli bir değişkene atadık. 
History değişkeni içinde sözlük halinde modelin her turdaki bilgileri tutulacak ve görselleştirme için kullanacağız.
"""

#%%
plt.figure(figsize=(10,6)) # GÖRSEL BOYUTU İNÇ CİNSİNDEN BELİRLENİYOR.
plt.subplot(1,2,1) # 1 SATIR 2 SÜTÜNDAN OLUŞAN PLOT IZGARASININ 1. PLOT'UNU ÇİZDİRİYORUZ.
sns.regplot(list(range(1,6)), history.history["acc"], label="ACC", color=sns.xkcd_rgb["medium green"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.subplot(1,2,2) # 1 SATIR 2 SÜTÜNDAN OLUŞAN PLOT IZGARASININ 2. PLOT'UNU ÇİZDİRİYORUZ.
sns.regplot(list(range(1,6)), history.history["loss"], label="LOSS", color= sns.xkcd_rgb["pale red"])
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.suptitle("Accuracy and Loss", size=20)
