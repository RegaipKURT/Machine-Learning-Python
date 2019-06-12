#keras içerisindeki inception hazır modeliyle resimlerden nesne ve hayvan tanıma
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

model = InceptionV3(weights='imagenet', include_top=True)
konum = str(input("\nLütfen fotoğrafların olduğu klasör yolunu giriniz: "))
dosyalar = os.listdir(konum)
for i in range(1, (len(dosyalar)+1)):
    img_path = "{}/{}".format(konum ,dosyalar[i-1])
    
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    
    features = model.predict(x)
    print("\n{} resmi için ilk 3 tahmin:\n".format(dosyalar[i-1]), decode_predictions(features, top = 3))
    
    plt.imshow(image.load_img(img_path))
    plt.title("{}".format(decode_predictions(features, top = 1)))
    plt.show()
