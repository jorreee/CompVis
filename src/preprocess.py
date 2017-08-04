# -*- coding: utf-8 -*-
import cv2
import numpy as np

def apply_filter_train(image):
    #Apply mediam bilateral filter
    image1 = cv2.medianBlur(image,9)
    image2 = cv2.bilateralFilter(image1,9,175,175)
    kernel = np.ones((100,100),np.uint8)
    #Apply tophat transform
    image3 = cv2.add(image2, cv2.morphologyEx(image2,cv2.MORPH_TOPHAT,kernel))
    kernel = np.ones((25,25),np.uint8)
    #Apply bottomhat transform
    image4 = cv2.subtract(image3, cv2.morphologyEx(image2,cv2.MORPH_BLACKHAT,kernel))
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(16,16))
    #Apply CLAHE
    image5 = clahe.apply(image4)
    
    return image5
    
def apply_canny(img_o,img):
    h1 = 10*2*2*1.5
    detection = cv2.Canny(img,h1/2,h1)
    result = np.zeros(img.shape)
    result[detection.astype(np.bool)]=1
    
    return result
    
def apply_laplacian(img):
    laplacian = cv2.Laplacian(img,cv2.CV_64F,ksize=11)
    
    return laplacian
    
def apply_sobel(img):
    img = cv2.GaussianBlur(img,(3,3),0)
    kersize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kersize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kersize)
    
    absx = cv2.convertScaleAbs(sobelx)
    absy = cv2.convertScaleAbs(sobely)
    
    combined = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    return combined
    
    
if __name__ == '__main__':
    img = cv2.imread("data/Radiographs/02.tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[550:1430,1150:1850]
    img_o = np.copy(img)
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width = img.shape;
    cv2.resizeWindow('image', width / 2, height /2)
    cv2.imshow('image',img);
    #cv2.waitKey(0)

    img = apply_filter_train(img)    
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width = img.shape;
    cv2.resizeWindow('image', width / 2, height /2)
    cv2.imshow('image',img);
    #cv2.waitKey(0)

    img = apply_sobel(img)  
    #img[img > 30] *= 2  
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width = img.shape;
    cv2.resizeWindow('image', width / 2, height /2)
    cv2.imshow('image',img);

    
