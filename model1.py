import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import  Sequential,Model
from keras.layers import  Conv2D, MaxPooling2D,InputLayer,UpSampling2D,BatchNormalization,Input
from keras_preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.callbacks import TensorBoard
from skimage.color import rgb2lab,lab2rgb,gray2rgb
from cv2 import resize
from keras.applications.vgg16 import VGG16
from skimage.io import imsave

from keras.applications.vgg16 import VGG16

#initializing VGG16 layer
vggmodel=VGG16()
newmodel=Sequential()

#taking 18 layer of VGG16
for i, layer in enumerate(vggmodel.layers):
    if i<19:
        newmodel.add(layer)
newmodel.summary()

for layer in newmodel.layers:
    layer.trainable=False

#path for image dataset
path = 'C:/Users/Acer/OneDrive/Desktop/major project/dataset/trainimg/'

#resizing the image since we are doing the relu activation function which activates between 0 and 1
train_datagen = ImageDataGenerator(rescale=1. / 255)
train= train_datagen.flow_from_directory(path, target_size=(224, 224),batch_size=5000,class_mode=None)
X_train, X_test = train_test_split(train[0], test_size=0.1,random_state=1)
len(X_train)

#saving the  train and test images
i=0
for i in range(len(X_train)):
    imsave('C:/Users/Acer/OneDrive/Desktop/major project/train/train%i.jpg'%i,X_train[i])
len(X_test)

i=0
for i in range(len(X_test)):
    imsave('C:/Users/Acer/OneDrive/Desktop/major project/test/test%i.jpg'%i,X_test[i])

#storing the L and ab channel
X =[]
Y =[]
for img in X_train:
  try:
      lab = rgb2lab(img)
      X.append(lab[:,:,0])
      Y.append(lab[:,:,1:] /128)
  except:
     print('error')
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,))
print(X.shape)
print(Y.shape)  

#extracting the VGG16 layer
vggfeatures = []
for i, sample in enumerate(X):
    sample = gray2rgb(sample)
    sample = sample.reshape((1,224,224,3))
    prediction = newmodel.predict(sample)
    print(prediction.shape)
    prediction = prediction.reshape((7,7,512))
    vggfeatures.append(prediction)
    vggfeatures = np.array(vggfeatures)
print(vggfeatures.shape)

#Encoder
encoder_input = Input(shape=(7, 7, 512,))
#Decoder
decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_input)
print(decoder_output.shape)
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
model = Model(inputs=encoder_input, outputs=decoder_output)
model.summary()

#compiling model
model.compile(optimizer='Adam', loss='mse' , metrics=['accuracy'])
history=model.fit(vggfeatures, Y,validation_split=0.1, verbose=1, epochs=200, batch_size=32) 

#Graph for validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Graph for validation accuracy
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'y', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#testing the model 
i=0
for i in range(len(X_test)):
    lab = rgb2lab(X_test[i])
    l= lab[:,:,0]
    imsave('C:/Users/Acer/OneDrive/Desktop/major project/testbw/test%i.jpg'%i, l)
    test = gray2rgb(l)
    tests= test.reshape((1,224,224,3)) 
    vggpred = newmodel.predict(tests)
    print(vggpred.shape)
    ab = model.predict(vggpred)
    ab = ab*128
    result= np.zeros((224, 224, 3))
    result[:,:,0] = l
    result[:,:,1:] = ab  
    result=lab2rgb(result)
    result=cv2.resize(result,(350,350))
    imsave('C:/Users/Acer/OneDrive/Desktop/major project/testcolor/test%i.jpg'%i, result)
    plt.imshow(lab2rgb(result))

i=0
for i in range(len(X_train)):
    lab = rgb2lab(X_train[i])
    l= lab[:,:,0]
    imsave('C:/Users/Acer/OneDrive/Desktop/major project/testbw/test%i.jpg'%i, l)
    test = gray2rgb(l)
    tests= test.reshape((1,224,224,3)) 
    vggpred = newmodel.predict(tests)
    print(vggpred.shape)
    ab = model.predict(vggpred)
    ab = ab*128
    result= np.zeros((224, 224, 3))
    result[:,:,0] = l
    result[:,:,1:] = ab  
    result=lab2rgb(result)
    result=cv2.resize(result,(350,350))
    imsave('C:/Users/Acer/OneDrive/Desktop/major project/traincolor/train%i.jpg'%i, result)
    plt.imshow(lab2rgb(result))

#saving model
import pickle
pickle.dump(newmodel,open('img_model2.pkl','wb'))
pickle.dump(model,open ('image_color2.pkl','wb'))

