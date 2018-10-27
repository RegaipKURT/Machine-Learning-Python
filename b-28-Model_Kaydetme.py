# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

url = "http://www.bilkav.com/wp-content/uploads/2018/03/satislar.csv"
veriler = pd.read_csv(url)

X = veriler.iloc[:,0:1].values
Y = veriler.iloc[:,1:].values

bolme = 0.33

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = bolme) 

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,Y_train) 
y_pred = lr.predict(X_test)
print (y_pred)

# burdan sonra eğittiğimiz modeli kaydedelim
import pickle
lineer_tahmin = "linear_model.kayit" 
pickle.dump(lr, open(lineer_tahmin,"wb")) #modeli kaydedecek
"""
DİKKAT ET: Kaydederken wb, yüklerken rb modunda açacağız dosyayı
"""

#kaydedilen dosyayı daha sonra bu komutla ortam farketmeksizin
#çağırıp kullanabiliriz. Hatta uzak sunucular üzerinden bile.
yuklenen = pickle.load(open(lineer_tahmin,"rb")) 
print (yuklenen.predict(X_test)) #X_test yerine kendi verini girmelisin başka bilgisayarda

import matplotlib.pyplot as plt

plt.scatter(Y_test,X_test,color = "green")
plt.plot(y_pred,X_test,color = "red")




