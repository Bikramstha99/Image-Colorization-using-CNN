import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imsave
import cv2
import pickle
import numpy as np
from skimage.color import rgb2lab,lab2rgb,gray2rgb

def getPrediction(filename):
    model1=pickle.load(open('img_model2.pkl','rb'))
    model2=pickle.load(open('image_color2.pkl','rb'))
    img_path='web pic/black and white image/'+filename
    test=np.asarray(Image.open(img_path).resize((224,224)))
    test1=cv2.resize(test,(350,350))
    imsave('C:/Users/Acer/OneDrive/Desktop/major project/static/savedbw/img.jpg', test1)

    
    if(test.shape==(224,224,3)):
        tests = test.reshape((1,224,224,3))  
        lab = rgb2lab(test)
        l= lab[:,:,0]
        #print(tests.shape)
        vggpred = model1.predict(tests)
        print(vggpred.shape)
        ab = model2.predict(vggpred)
        print(ab.shape)
        ab = ab*128
        result= np.zeros((224, 224, 3))
        result[:,:,0] = l
        result[:,:,1:] = ab 
        color=lab2rgb(result) 
        color=cv2.resize(color,(350,350))
        imsave('C:/Users/Acer/OneDrive/Desktop/major project/static/savedcolor/img1.jpg', color)
        return color
    else:
        tests = gray2rgb(test)
        tests = tests.reshape((1,224,224,3))  
        vggpred = model1.predict(tests)
        print(vggpred.shape)
        ab = model2.predict(vggpred)
        print(ab.shape)
        ab = ab*128
        result= np.zeros((224, 224, 3))
        result[:,:,0] = test
        result[:,:,1:] = ab 
        color=lab2rgb(result) 
        color=cv2.resize(color,(350,350))
        imsave('C:/Users/Acer/OneDrive/Desktop/major project/static/savedcolor/img1.jpg', color)
        return color

        


#test_prediction = getPrediction('black.jpg')
