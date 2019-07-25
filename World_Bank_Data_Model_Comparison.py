import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import SGD

veri = pd.read_csv("datas/WB.csv") #WB.csv dosyası repository içinde var, indirip kullanılabilir.
print(veri.head())

x = veri["Date"].values.reshape(-1,1)
y = veri["Value"].values.reshape(-1,1)
y = np.ravel(y)
print(x.shape, y.shape)

le = LabelEncoder()
x = le.fit_transform(x)
x = x.reshape(-1,1)

print(x.shape)
print(x)

lr = LinearRegression()
lr.fit(x,y)

y_pred = lr.predict(x)
print("Başlangıç: ", lr.intercept_)
y_pred_lr = lr.predict([[29]])
print(y_pred_lr)

pf = PolynomialFeatures(degree=2)
x2 = pf.fit_transform(x)

lr.fit(x2, y)
y_log = lr.predict(x2)
print(x2)

y_pred_pr = lr.predict([[1,29,841]])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33)
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)

model = Sequential()
model.add(Dense(32, kernel_initializer="glorot_uniform", activation="relu"))
model.add(Dense(32, activation="tanh"))
model.add(Dense(30, activation="relu"))
model.add(Dense(1))

model.compile(batch_size=1, optimizer=Adam(lr=0.03), loss="mse")
model.fit(x_train, y_train, epochs=20000)

y_pred_ker = model.predict([[29]])
y_pred_k = model.predict(x)

plt.figure(figsize=(12.8, 10.24), dpi=100, facecolor='w', edgecolor='k')
plt.scatter(x,y, label="Original Data")
plt.plot(x,y)
plt.scatter(29,y_pred_lr, color="red", label="Linear Predict 2019 Growth Rate", marker="X")
plt.scatter(29,y_pred_pr, color="black", label="Polynomial Predict 2019 Growth Rate", marker="X")
plt.scatter(29,y_pred_ker, color="purple", label="Keras Predict 2019 Growth Rate", marker="X")
plt.plot(x, y_pred, color="red", label="Linear Regression")
plt.plot(x, y_log, color="black", label="Polynomial Regression")
plt.plot(x, y_pred_k, color="purple", label="Keras")

plt.title("Dünya Bankası\nBüyüme Verisi")
plt.xticks(x, veri["Date"], rotation=90)
plt.ylabel("Baz Yıl Tabanlı Büyüme Verisi")
plt.xlabel("TARİH")
plt.legend()

plt.savefig("Dünya Büyüme Modeli Karşılaştırmalı.png")
plt.show()
