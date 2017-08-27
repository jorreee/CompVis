# -*- coding: utf-8 -*-
import cv2
import numpy as np

# Applies the filter train as discussed by Huang et al in
#  "Noise Removal and Contrast Enhancement for X-Ray Images".
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

# Applies the Canny operator. Not used in the rest of this project.    
def apply_canny(img_o,img):
    h1 = 10*2*2*1.5
    detection = cv2.Canny(img,h1/2,h1)
    result = np.zeros(img.shape)
    result[detection.astype(np.bool)]=1
    
    return result

# Applies the Laplacian operator. Not used in the rest of this project.        
def apply_laplacian(img):
    laplacian = cv2.Laplacian(img,cv2.CV_64F,ksize=11)
    
    return laplacian

# Applies the Sobel operator to detect edges.     
def apply_sobel(img):
    img = cv2.GaussianBlur(img,(3,3),0)
    kersize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kersize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kersize)
    
    absx = cv2.convertScaleAbs(sobelx)
    absy = cv2.convertScaleAbs(sobely)
    
    combined = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    return combined
    

    
