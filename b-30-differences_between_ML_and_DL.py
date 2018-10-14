from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
X=sc.fit_transform(x)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
lda = LinearDiscriminantAnalysis(n_components=1) #bir boyuta indirdik
X_train2 = lda.fit_transform(X_train, y_train)
X_test2 = lda.fit_transform(X_test, y_test)
X=lda.fit_transform(x,y)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
#PCA uygulanmamış veri
lr2 = LogisticRegression()
lr2.fit(X_train2,y_train)

#Predictions
y_pred = lr.predict(X_test)
y_pred2 = lr2.predict(X_test2)
print (X.shape)

from keras.utils import to_categorical
y = to_categorical(y, num_classes=3)
y_test2 = to_categorical(y_test, num_classes=3)

from keras.layers import Dense
from keras.models import Sequential

model = Sequential()

model.add(Dense(10, kernel_initializer='glorot_uniform', activation='relu', input_dim=1))
model.add(Dense(8,activation='tanh'))
model.add(Dense(6,activation='tanh'))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer="SGD",loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(X,y,batch_size=5,epochs=200)

y_pred3 = model.predict(X_test2) #keras prediction

from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

print ("\nOriginal Prediction Matrix:")
cm1 = confusion_matrix(y_test, y_pred)
print (cm1)
print ("\nLDA Applied Prediction:")
cm2 = confusion_matrix(y_test, y_pred2)
print (cm2)
cm1 = confusion_matrix(y_test, y_pred)
r2_1 = r2_score(y_test,y_pred)
print("\nOrjinal R2 score:", r2_1)
r2_2 = r2_score(y_test,y_pred2)
print("LDA Applied R2 score:", r2_2)
r2_3 = r2_score(y_test2,y_pred3)
print ("Keras R2 score:",r2_3)






