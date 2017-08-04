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
# Takes a column vector and interprets it as all x coordinates followed by all y coordinates, then draws a line between the points on the given image
def draw_contour(img,lm,color=(255,0,0),thicc=1):
#    for ind in range(lengthn-1):
	lpts = np.reshape(lm,((len/2) - 1,1),'F')
    for ind in range(len(lpts)):
        if (ind+1) % 40 == 0:
            continue
        else:
            cv2.line(img,
                (int(float(lpts[ind,0])),int(float(lpts[ind,1]))),
                (int(float(lpts[ind+1,0])),int(float(lpts[ind+1,1]))),
                color,
                thicc);
    
    return None
    
# Takes a matrix with landmarks as columns and draws them all on the given image
def draw_all_contours(img,ls):
    for lm in ls.T:
        draw_landmark(img,lm.T)
    
    return None
	
def convert_to_image_space(ls):
    lss = np.copy(ls)
    for el in lss:
        for i in range(0,len(el)):
            el[i,0] = el[i] * 503 + 250
	return lss

# Takes a matrix with aligned landmarks as column. Applies a scaling and translation transformation to them to be visible in image space and then draws the transformed landmarks on the given image
def draw_aligned_contours(img,ls):
	lst = np.copy(ls).T # alle lms zijn nu de rijen
	for i in range(lst.shape[0]): # ga over alle rijen
		lst[i] = convert_to_image_space(lst[i])
    draw_all_contours(img,lst.T) # zet lms terug als columns en teken
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