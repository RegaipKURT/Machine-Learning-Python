import numpy as np
import matplotlib.pyplot as plt

# ilk önce normal bir numpy dizisi oluşturalım
dizi1 = np.random.normal(100, scale=10, size=100)

#KULLANACAĞIMIZ FONKSİYONLAR

#1. FONKSİYON MEVCUT DEĞERDEN MİNİMUM DEĞERİ ÇIKARIP MAX VE MİN ARASINDAKİ FARKA BÖLECEK.
# VERİLER 0 İLE 1 ARASINA SIKIŞTIRILMIŞ OLACAK.
def normalize(liste):
    yeniliste = []
    for i in liste:
        a = (i-min(liste))/(max(liste)-min(liste))
        yeniliste.append(a)
    return yeniliste


#2. FONKSİYON MEVCUT DEĞERDEN ORTALAMA DEĞERİ ÇIKARIP MAX VE ORTALAMA ARASINDAKİ FARKA BÖLECEK.
# VERİLER -1 İLE 1 ARASINA SIKIŞTIRILMIŞ OLACAK.
def standardize(liste):
    yeniliste = []
    for i in liste:
        a = (i - np.mean(liste)) / (max(liste) - np.mean(liste))
        yeniliste.append(a)
    return yeniliste


#3. FONKSİYON MEVCUT DEĞERDEN ORTALAMA DEĞERİ ÇIKARIP MAX VE MİN ARASINDAKİ FARKA BÖLECEK.
# VERİLER 0.5 İLE -0.5 ARASINA SIKIŞTIRILMIŞ OLACAK.
def standardize2(liste):
    yeniliste = []
    for i in liste:
        a = (i - np.mean(liste)) / (max(liste) - np.min(liste))
        yeniliste.append(a)
    return yeniliste

# GÖRSELLEŞTİRME KISMI
# LİSTENİN ORJİNAL HALİ
plt.subplot(2, 2, 1) # 2 SATIR 2 SÜTUNDAN OLUŞAN PLOTUN 1.Sİ
plt.plot(dizi1, label="Normal Liste")
plt.title("Değerler")
plt.ylabel("Aralıklar")
plt.legend()

# 1. METOT İLE OLUŞTURULAN LİSTE
plt.subplot(2, 2, 2) # 2 SATIR 2 SÜTUNDAN OLUŞAN PLOTUN 2.Sİ
plt.title("Değerler")
plt.plot(normalize(dizi1), label="Normalize Liste")
plt.legend()

# 2. METOT İLE OLUŞTURULAN LİSTE
plt.subplot(2, 2, 3)  # 2 SATIR 2 SÜTUNDAN OLUŞAN PLOTUN 3.SÜ
plt.ylabel("Aralıklar")
plt.plot(standardize(dizi1), label="Standardize Liste")
plt.legend()

# 3. METOT İLE OLUŞTURULAN LİSTE
plt.subplot(2, 2, 4) # 2 SATIR 2 SÜTUNDAN OLUŞAN PLOTUN 4.SÜ
plt.plot(standardize2(dizi1), label="Standardize 2. Liste")
plt.legend()
plt.show()

# BAŞKA FARKLI METOTLAR DA YAZILABİLİR.