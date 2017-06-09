# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
from numpy import linalg as LA
import fnmatch

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
        cv2.line(img,(int(float(lpts[ind,0])),int(float(lpts[ind,1]))),(int(float(lpts[ind+1,0])),int(float(lpts[ind+1,1]))),(255,0,0),2);
    
    cv2.line(img,(int(float(lpts[39,0])),int(float(lpts[39,1]))),(int(float(lpts[0,0])),int(float(lpts[0,1]))),(255,0,0),2);
    
    return None
    
def draw_all_landmarks(img,ls):
    for lm in ls:
        draw_landmark(img,lm)
    
    return None
    
def align(ls):
    # Calculate centroids
    centroids = map(lambda lpts: extract_centroid(lpts),ls)
    
    # Translate by centroid difference
    centref = centroids[0]
    centdiff = map(lambda e: centref - e, centroids)
    for ind in range(len(centroids)):
        ls[ind] = ls[ind] + centdiff[ind]
          
    # Calculate scale and rotate
    lsflat = map(lambda e: e.flatten(),ls)
    x1 = lsflat[0]
    x1norm2 = LA.norm(x1) * LA.norm(x1)
    for ind in range(len(lsflat)):
        x2 = lsflat[ind]
        a = np.dot(x1,x2) / x1norm2
        b = 0
        for j in range(0,len(x1),2):
            b += x1[j] * x2[j+1] - x1[j+1] * x2[j]
        b /= x1norm2
        s = a * a + b * b
        theta = np.arctan(b/a)
        srm = np.array([[s * np.cos(theta), -s * np.sin(theta)], [s * np.sin(theta), s * np.cos(theta)]])
        for k in range(0,40):
            nr = np.dot(srm,ls[ind][k,:])
            ls[ind][k,0] = nr[0]
            ls[ind][k,1] = nr[1]
   
    return ls
    
def extract_centroid(lpts):
    return np.array([np.mean(lpts[:,0]),np.mean(lpts[:,1])]) 
    
if __name__ == '__main__':
    
    #print read_landmark('data/Landmarks/original/landmarks1-1.txt')

    img = cv2.imread("data/Radiographs/01.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    #tmp = open('data/Landmarks/original/landmarks1-1.txt').readlines();
    #for ind in range(0, len(tmp), 2):
    #    if (ind + 2 >= len(tmp)):
    #        cv2.line(img,(int(float(tmp[ind].strip())),int(float(tmp[ind +1].strip()))),(int(float(tmp[0].strip())),int(float(tmp[1].strip()))),(255,0,0),2);
    #    else :
    #        cv2.line(img,(int(float(tmp[ind].strip())),int(float(tmp[ind +1].strip()))),(int(float(tmp[ind +2].strip())),int(float(tmp[ind +3].strip()))),(255,0,0),2);
    lm = read_all_landmarks_by_tooth(2)
    lm = align(lm)
    draw_all_landmarks(img,lm)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width, _ = img.shape;
    cv2.resizeWindow('image', width / 2, height /2)
    cv2.imshow('image',img);

    
