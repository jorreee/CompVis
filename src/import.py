# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
from numpy import linalg as LA
import fnmatch
import math


def read_landmark(lm_file):
    return np.reshape(map(lambda x: float(x.strip()),open(lm_file).readlines()),(40,2))
    
def read_mirrored_landmark(lm_file):
    pts = np.reshape(map(lambda x: float(x.strip()),open(lm_file).readlines()),(40,2))
    for i in range(40):
        pts[i,0] = 3022 + pts[i,0]
    return pts

def read_all_landmarks_by_img(imgi):
    ls = []
    for i in range(8):
        ls.append(read_landmark('data/Landmarks/landmarks'+str(imgi)+'-'+str(i+1)+'.txt'))
    return ls

def read_all_landmarks_by_tooth(toothi):
    ls = []
    for i in range(14):
        ls.append(read_landmark('data/Landmarks/landmarks'+str(i+1)+'-'+str(toothi)+'.txt'))
    for i in range(14):
        ls.append(read_mirrored_landmark('data/Landmarks/landmarks'+str(i+15)+'-'+str(toothi)+'.txt'))
    return ls
    
def draw_landmark(img,lpts):
    for ind in range(39):
        cv2.line(img,(int(float(lpts[ind,0])),int(float(lpts[ind,1]))),(int(float(lpts[ind+1,0])),int(float(lpts[ind+1,1]))),(255,0,0),1);
    
    cv2.line(img,(int(float(lpts[39,0])),int(float(lpts[39,1]))),(int(float(lpts[0,0])),int(float(lpts[0,1]))),(255,0,0),1);
    
    return None
    
def draw_all_landmarks(img,ls):
    for lm in ls:
        draw_landmark(img,lm)
    
    return None
    
def draw_aligned_landmarks(img,ls):
    lss = ls
    for el in lss:
        for i in range(0,len(el)):
            el[i,0] = el[i,0] * 503 + 250
            el[i,1] = el[i,1] * 503 + 250
    for lm in lss:
        draw_landmark(img,lm)
    return None
    
def procrustes(ls):
    
    #Translate landmarks to origin
    ls = center_orig_all(ls)
    
    #Flatten landmark list
    lsflat = map(lambda e: e.flatten(),ls)
    
    #Choose first example for initial mean estimate and rescale
    x0 = lsflat[0]
    x0 = normalize(x0)
    mean = x0

    while True:
        
        # Align all landmarks with mean
        for i in range(0,len(lsflat)) :
            lsflat[i] = align(lsflat[i], mean)
                    
        # Re-estimate the mean
        new_mean = estimate_mean(lsflat)
        
        # Center new mean
        new_mean = center_orig(np.reshape(new_mean,(40,2))).flatten()
        # Normalize and align new mean to initial mean
        new_mean = normalize(align(new_mean,x0))
        # Re-center new mean
        new_mean = center_orig(np.reshape(new_mean,(40,2))).flatten()

        # Check for convergence
        if (mean - new_mean < 0.000001).all():
            break
        else:
            mean = new_mean
    
    # Reshape the result
    ls = map(lambda e: np.reshape(e,(40,2)),lsflat)
    mean = np.reshape(mean,(40,2))
    return [ls,mean]
    

def extract_centroid(lpts):
    return np.array([np.mean(lpts[:,0]),np.mean(lpts[:,1])]) 

# Center one landmark to origin
def center_orig(mark):
    centroid = extract_centroid(mark)
    mark = mark - centroid
    return mark
    
    
# Center landmarks ls to origin
def center_orig_all(ls):
    return map(lambda lpts: center_orig(lpts),ls)
    
def normalize(v):
    norm=LA.norm(v)
    if norm==0: 
       return v
    return v/norm

# Estimates the mean landmark of a set of landmarks for a tooth
def estimate_mean(lsflat):
    mean = np.zeros(len(lsflat[0]))
    for mark in lsflat:
        mean += mark
    return mean/(len(lsflat[0]))
    

#Aligns shape x1 to x2
def align(x1,x2):
    x1norm2 = np.dot(x1,x1)
    a = np.dot(x1,x2) / x1norm2
    b = 0
    for j in range(0,len(x1),2):
        b += x1[j] * x2[j+1] - x1[j+1] * x2[j]
    b /= x1norm2
    s = math.sqrt(a * a + b * b)
    theta = np.arctan(b/a)
    srm = np.array([[s * np.cos(theta), -s * np.sin(theta)], [s * np.sin(theta), s * np.cos(theta)]])
    for k in range(0,len(x1),2):
        nr = np.dot(srm,[x1[k],x1[k+1]])
        x1[k] = nr[0]
        x1[k+1] = nr[1]
    
    return x1
    

    
if __name__ == '__main__':
    
    # Dental radiograph inladen
    #img = cv2.imread("data/Radiographs/01.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    
    # Lege witte image maken voor procrustes afbeelding
    img = np.zeros((512,512,3), np.uint8)
    img[:] = (255, 255, 255)
    
    #tmp = open('data/Landmarks/original/landmarks1-1.txt').readlines();
    #for ind in range(0, len(tmp), 2):
    #    if (ind + 2 >= len(tmp)):
    #        cv2.line(img,(int(float(tmp[ind].strip())),int(float(tmp[ind +1].strip()))),(int(float(tmp[0].strip())),int(float(tmp[1].strip()))),(255,0,0),2);
    #    else :
    #        cv2.line(img,(int(float(tmp[ind].strip())),int(float(tmp[ind +1].strip()))),(int(float(tmp[ind +2].strip())),int(float(tmp[ind +3].strip()))),(255,0,0),2);
    lm = read_all_landmarks_by_tooth(1)
    lm,mean = procrustes(lm)
    print mean
    draw_aligned_landmarks(img,lm)
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#    height, width, _ = img.shape;
#    cv2.resizeWindow('image', width / 2, height /2)
    cv2.imshow('image',img);

    
