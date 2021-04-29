import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Multiply
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.metrics import RootMeanSquaredError as rmse
from keras.activations import relu
from keras.optimizers import Adam
from tensorflow.python.keras.engine.training import Model
from keras.utils import plot_model


'''
This file created to show how to use merge layers. (like multiplication layer all of them are used with same method)
We will multiply 2 numbers and then we are going to use 2 different models to predict result of the multiplication.
First model is gonna use Multiply layer and second is going to consist from dense layers.
Since we have multiplied two numbers we used Multiply layer.
You can use other merge layers like Add or Subtract to use in your own projects.

We will discover success of two methods will be different.
Because it is is much more usefull to use Multiply layer when multiply numbers (no surprise).

We also used multiple input layers at the model which contains multiply layer.

When multiplication layer is used train time also be reduced.
'''

#Function to Create Dataset
def create_multiplication_file(filename, start, stop):
    with open(filename, "w") as f:
        f.writelines("first,second,result\n")
        for i in range(-start, stop+1):
            for j in range(start, stop+1):
                f.writelines(str(i)+ "," + str(j) + "," + str(i*j) + "\n")


create_multiplication_file(filename="f.csv", start=10, stop=100)

#Load Dataset
d = pd.read_csv("f.csv", index_col=None)
x = d.iloc[:,:2]
y = d.iloc[:,2]

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=True)
print(X_train)
xt1 = X_train["first"]
xt2 = X_train["second"]


#model 1 with multiplication layer
inp1 = Input((1,))
inp2 = Input((1,))
m = Multiply()([inp1, inp2])
d1 = Dense(32, activation=relu)(m)
d2 = Dense(32, activation=relu)(d1)
out = Dense(1)(d2)
model = Model([inp1, inp2], out)


#model 2 without multiplication layer. Only Dense Layers.
model2 = Sequential([
    Dense(32, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])

#early stopping
es1 = EarlyStopping(monitor='val_rmse', mode='min', verbose=1, patience=1)
es2 = EarlyStopping(monitor='val_rmse', mode='min', verbose=1, patience=1)


epochs = 500

#compile models
model.compile(optimizer=Adam(), loss="mse", metrics=[rmse(name="rmse")])
history = model.fit(x=[xt1, xt2], y=y_train, epochs=epochs, validation_data=([X_test["first"],X_test["second"]], y_test), callbacks=[es1])

model2.compile(optimizer=Adam(), loss="mse", metrics=[rmse(name="rmse")])
history2 = model2.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es2])

#predictions
y_preds = model.predict([X_test["first"],X_test["second"]])
y_preds2 = model2.predict(X_test)


#PLOT RESULTS

#styles
sns.set_style("darkgrid")

#MODEL1 RESULTS
plt.figure(figsize=(16,12))
plt.subplot(221)
sns.lineplot(range(0, es1.stopped_epoch+1) if es1.stopped_epoch != 0 else range(0, epochs), 
             history.history["rmse"],
             label="train rmse 1")
sns.lineplot(range(0, es1.stopped_epoch+1) if es1.stopped_epoch != 0 else range(0,epochs), 
             history.history["val_rmse"], 
             label="val rmse 1")
plt.title("Errors")
plt.legend()
plt.subplot(222)
sns.lineplot(range(0, y_test.shape[0]), y_test,
             label="test actual 1")
sns.lineplot(range(0, y_test.shape[0]), y_preds[:,0],
             label="test preds 1")
plt.legend()
plt.title("Predictions")


#MODEL2 RESULTS
plt.subplot(223)
sns.lineplot(range(0, es2.stopped_epoch+1) if es2.stopped_epoch != 0 else range(0, epochs), 
             history2.history["rmse"],
             label="train rmse 2")
sns.lineplot(range(0, es2.stopped_epoch+1) if es2.stopped_epoch != 0 else range(0,epochs), 
             history2.history["val_rmse"], 
             label="val rmse 2")
plt.legend()
plt.subplot(224)
sns.lineplot(range(0, y_test.shape[0]), y_test,
             label="test actual 2")
sns.lineplot(range(0, y_test.shape[0]), y_preds2[:,0],
             label="test preds 2")
plt.legend()

#SHOW PLOT
plt.suptitle(f"RESULTS\nRMSE when Multiplication layer used: {history.history['val_rmse'][-1]}\nRMSE when Multiplication layer not used: {history2.history['val_rmse'][-1]}")
plt.savefig("results.png", dpi=100)

plt.show()

#RESULTS TO DATAFRAME
results = pd.DataFrame([y_preds[:,0], y_preds2, y_test, X_test.iloc[:,0], X_test.iloc[:,1]]).T
results.columns = ["Model with Multiplication", "Model without Multiplication", "Actual Result", "First Value", "Second Value"]
results["Model without Multiplication"] = results["Model without Multiplication"].astype(float)
print(results)
print("Model with Multiplication RMSE: ", model.evaluate(x=[X_test["first"],X_test["second"]], y=y_test)[1])
print("Model without Multiplication RMSE: ", model2.evaluate(x=X_test, y=y_test)[1])

plot_model(model, to_file="model1.png")
plot_model(model2, to_file="model2.png")

data1 = plt.imread("model1.png")
data2 = plt.imread("model2.png")

plt.figure(figsize=(16,12))
plt.subplot(121)
plt.imshow(data1)
plt.title("Model with Multiply Layer")
plt.subplot(122)
plt.imshow(data2)
plt.title("Model with Dense Layers Only")
plt.savefig("models_graph.png", dpi=100)
plt.show()
