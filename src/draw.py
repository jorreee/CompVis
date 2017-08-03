# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
from numpy import linalg as LA
import sklearn.decomposition as skldecomp
import fnmatch
import math

lengthn = 160
def draw_landmark(img,lpts,color=(255,0,0),thicc=1):
    lpts = np.reshape(lpts,(lpts.size / 2, 2),'F')
    for ind in range(lengthn-1):
        if (ind+1) % 40 == 0:
            continue
        else:
            cv2.line(img,
                (int(float(lpts[ind,0])),int(float(lpts[ind,1]))),
                (int(float(lpts[ind+1,0])),int(float(lpts[ind+1,1]))),
                color,
                thicc);
    
    # cv2.line(img,(int(float(lpts[lengthn-1,0])),int(float(lpts[lengthn-1,1]))),(int(float(lpts[0,0])),int(float(lpts[0,1]))),color,1);
    
    return None
    
def draw_all_landmarks(img,ls):
    for lm in ls.T:
        draw_landmark(img,lm.T)
    
    return None
    
def draw_aligned_landmarks(img,ls):
    lss = ls
    for el in lss:
        for i in range(0,len(el)):
            el[i,0] = el[i,0] * 503 * np.sqrt(lengthn/40) + 250
            el[i,1] = el[i,1] * 503 * np.sqrt(lengthn/40) + 250
    for i in range(len(lss)):
        if i == 0:
            draw_landmark(img,lss[i],color=(0,0,0),thicc=2)
        else:
            draw_landmark(img,lss[i],color=(200,200,0),thicc=2)
    return None

def draw_pca_reconstruction(img,lms):
    lss = lms
    lsss = map(lambda el: el.reshape(2,lengthn), lss)
    for k in range(len(lsss)):
        el = lsss[k].T
        for i in range(lengthn):
            el[i,0] = el[i,0] * 503 + 250
            el[i,1] = el[i,1] * 503 + 250
        draw_landmark(img,el)
    return None