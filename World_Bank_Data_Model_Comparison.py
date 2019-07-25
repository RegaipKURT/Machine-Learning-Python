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

veri = pd.read_csv("datas/WB.csv")
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

y_pred_lr = lr.predict([[32]])
print(y_pred_lr)

pf = PolynomialFeatures(degree=2)
x2 = pf.fit_transform(x)

lr.fit(x2, y)
y_log = lr.predict(x2)
print(x2)

y_pred_pr = lr.predict([[1,32,1024]])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33)

model = Sequential()
model.add(Dense(32, kernel_initializer="glorot_uniform", activation="relu"))
model.add(Dense(32, activation="tanh"))
model.add(Dense(30, activation="relu"))
model.add(Dense(1))

model.compile(batch_size=1, optimizer=Adam(lr=0.001), loss="mse")

model.fit(x_train, y_train, epochs=15000)
y_pred_ker = model.predict([[32]])
y_pred_k = model.predict(x)

plt.scatter(x,y, label="Original Data")
plt.scatter(32,y_pred_lr, color="purple", label="Linear Predict of 32", marker="X")
plt.scatter(32,y_pred_pr, color="black", label="Polynomial Predict of 32", marker="X")
plt.scatter(32,y_pred_ker, color="green", label="Keras NN Predict of 32", marker="X")
plt.plot(x, y_pred, color="red", label="Linear Regression")
plt.plot(x, y_log, color="black", label="Polynomial Regression")
plt.plot(x, y_pred_k, color="green", label="Keras")
plt.legend()
plt.show()