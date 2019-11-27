"""
Author: regkrt
"""
import numpy as np
import matplotlib.pyplot as plt

def dummy(parameter):
    dummy_parameter = parameter + 5
    return dummy_parameter

#lets began to initialize parameters and let's create a matrix of params
def initialize_parameters(dimension):
    #initializing weight values
    weight = np.full((dimension,1),0.01)
    # we defined weights matrix as w.
    # w is a matrix that is containing [dimension-1] size
    # our matrix's all values specified as 0.01
    # i mean our first w values is 0.01
    bias = 0.0
    """our bias's first value is 0. Pay attention, bias is float!!!"""
    return weight, bias
"""
#testing initialize function:
w, b = initialize_parameters(4096)
print (w,"\n", b)
print (w.shape)
"""

#creating a sigmoid function for treshold values
"""
z değeri: ağırlıklarla verileri çarpıp b değeri ile toplanarak bulunur.
z = np.dot(w.T, x_train) + b # this is the formule of z value
# yani; z = (ağırlıklar amtirisini transpozu * x_train) + b
#ağırlıkların transpozunu aldık matris çarpımı yapmak mümkün olsun diye
z will be a parameters of our sigmoid function and will return a prediction as a probalility
z sigmoid fonksiyona girecek ve 0 ile 1 arasında bir ihtimal döndürecek.
"""

def sigmoid(z):
    y_pred = 1 / (1+np.exp(-z))
    return y_pred

#forward and backward propagation process
"""
Aşağıda ileri ve geri yayılım  aşamalarını tamamlayıp  weight ve bias 
değerlerini güncellemek için bir fonksiyon yazdım. (güncelleme sonraki fonksiyonda)
loss(kayıp) ve cost(maliyet) fonksiyonlarının formüllerini yazdım ve sonra
weight ve bias değerlerinin türevlerini aldım.(maliyet(cost) fonksiyonuna göre türevleri)
"""
def forward_backward_propagation(weight, bias, x_train, y_train):
    
    #forward propagation
    z = np.dot(weight.T, x_train) + bias
    y_pred = sigmoid(z)
    loss = -y_train*np.log(y_pred) - (1-y_train)*np.log(1-y_pred)
    cost = (np.sum(loss)) / x_train.shape[1]
    
    #backward propagation
    derivative_weight = (np.dot(x_train,((y_pred-y_train).T)))/x_train.shape[1]
    #sonunda x_train'in boyutuna böldük ki ölçekleme ile ortalama bir değer alalım.
    #we divided to x_train shape at least for scaling
    derivative_bias = np.sum(y_pred-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    # we derived of weight and bias for updating parameters
    """
    gradients eğim demektir yani türev aslında. bu sözlük metodunun içinde
    ağırlık ve bias değerinin türevi bulunacak ve biz daha sonra bu sözlüğün
    içinden bu değerleri çağrıp, learning rate ile çarparak ilk baştaki
    weight ve bias değerimizden çıkaracaz ve yeni wwight ve bias değerini 
    bulmuş olacağız. formül: b_yeni = b_eski - (learning_rate*b_eskinin_türevi)
    Unutmamamız gereken şey türevler maliyet fonksiyonuna göre alınıyor.
    Çünkü biz maliyeti minimize etmek istiyoruz. Peki maliyetimiz neydi?:
    bütün hata değerlerinin toplamıydı, yani hatayı minimize edeceğiz.
    """
    return cost, gradients

#now let's write a functin for update parameters
#şimdi ağırlıkları ve bias değerini güncellemek için bir fonksiyon yazalım
def update(weight, bias, x_train, y_train, learning_rate, number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    #bu listeleri görselleştirme için yaptık asıl işle ilgili değil.
    #number_of_iteration epoch değeri veya kaç kere güncelleme yapılacağıdır.
    # updating parameters as number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(weight,bias,x_train,y_train)
        cost_list.append(cost)
        # lets update the weight and bias
        weight = weight - learning_rate * gradients["derivative_weight"]
        bias = bias - learning_rate * gradients["derivative_bias"]
        #ileri-geri yayılım fonksiyonunda tarif ettiğimiz işlemi yaptık.
        # 10 turda bir cost list içine maliyeti ekleyelim 
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    # we updated weights and bias
    parameters = {"weight": weight,"bias": bias}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

#prediction function
#artık değerler güncellendiğine ve hatayı mininum hale getirdiğimize göre
#bir tahmin yapıp sonuca bakalım
def predict(weight,bias,x_test):
    """
    Burada yaptığımız şey sigmoid fonksiyondan gelen z değerini alarak
    olasılığına göre karar vermek ve bir tahmin sonucu döndürmek.
    z 0.5 ten büyükse 1 değilse 0 kabul edeceğiz. Sonuç olarak da 1 veya 0
    yani doğru tahmin ya da yanlış tahmin diyeceğiz buna.
    """
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(weight.T,x_test)+bias)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    #z 0.5 ten büyükse 1 küçükse 0 döndürelim
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    return Y_prediction

def log_reg(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    """
    Burada aslında önceki fonksiyonlarımızı çalıştırıp bir doğruluk oranı oluşturacağız.
    
    """
    # initialize (ilklendirme; ilk weight ve bias değerleri)
    dimension =  x_train.shape[0]
    weight,bias = initialize_parameters(dimension)
    
    parameters, gradients, cost_list = update(weight, bias, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    # Print train/test Errors
    # ne kadar doğru bir tahmin yaptığımıza bakıyoruz.
    # orjinal y değeri le tahimin edilen y değerinin yüzdesini alıyoruz aslında
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

#şimdi kendi verimiz üzerinden bir deneme yapalım.
# let's test our codes 
"""
buradan sonra artık sklearn ile işaret dili verisini alarak
kendi modelimize koyup sonuçları görmek üzerine. 
""" 
# '../input/Sign-language-digits-dataset/X.npy'
# '../input/Sign-language-digits-dataset/Y.npy'
x_l = np.load('veriler/SLD/X.npy')
Y_l = np.load('veriler/SLD/Y.npy')
img_size = 64
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
zero = np.zeros(205)
one = np.ones(205)
Y = np.concatenate((zero, one), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=0)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]
X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)
x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

# you can take x_train, x_test and y_train, y_test values via your data using sklearn 
log_reg(x_train, y_train, x_test, y_test, learning_rate = 0.01, num_iterations = 200)

"""
Bütün bu yaptıklarımızdan sonra veri setimiz üzerinde gayet güzel bir tahmin
sonucuna ulaştık. Tabi grafikten de göreceksiniz ki hata bir yerden sonra 
artık düşmeyecek. Bu gayet doğaldır. Eğer yüzde yüz olsa bu ezberlemek demektir.
Zaten bu yüzden train ve test acuuracy'i birlikte yazdırdık ki bunu inceleyip
birbirlerine yakın oranlarda mı diye bakalım.
"""



















