# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
import scipy.signal as scisig
import scipy.ndimage.morphology as morf

def apply_filter_train(image):
    image1 = cv2.medianBlur(img,9)
    image2 = cv2.bilateralFilter(image1,9,175,175)
    kernel = np.ones((100,100),np.uint8)
    image3 = cv2.add(image2, cv2.morphologyEx(image2,cv2.MORPH_TOPHAT,kernel))
    kernel = np.ones((25,25),np.uint8)
    image4 = cv2.subtract(image3, cv2.morphologyEx(image2,cv2.MORPH_BLACKHAT,kernel))
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(16,16))
    image5 = clahe.apply(image4)
    
    return image5
    

    
if __name__ == '__main__':
    img = cv2.imread("data/Radiographs/04.tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[550:1430,1150:1850]
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width = img.shape;
    cv2.resizeWindow('image', width / 2, height /2)
    cv2.imshow('image',img);
    cv2.waitKey(0)

    img = apply_filter_train(img)    
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width = img.shape;
    cv2.resizeWindow('image', width / 2, height /2)
    cv2.imshow('image',img);

    
