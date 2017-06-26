# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
import scipy.signal as scisig
import scipy.ndimage.morphology as morf
import landmarks as lm
    
def draw_initial_landmarks(img,ls):
    lss = ls
    scalen = 900
    tx = 360
    ty = 310
    for el in lss:
        for i in range(0,len(el)):
            el[i,0] = el[i,0] * scalen * np.sqrt(lm.lengthn/40) + tx
            el[i,1] = el[i,1] * scalen * np.sqrt(lm.lengthn/40) + ty
    for i in range(len(lss)):
        if i == 0:
            lm.draw_landmark(img,lss[i],color=(0,0,0),thicc=2)
        else:
            lm.draw_landmark(img,lss[i],color=(200,200,0),thicc=2)
    return lss
    
if __name__ == '__main__':
    img = cv2.imread("data/Radiographs/05.tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[550:1430,1150:1850]
    
    lms = lm.read_all_landmarks_by_orientation(0)
    lms,mean = lm.procrustes(lms)
    init = draw_initial_landmarks(img,[mean])
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width = img.shape;
    cv2.resizeWindow('image', width / 2, height /2)
    cv2.imshow('image',img);

    
