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

def read_all_landmarks_by_orientation(toothies):
    ls = []
    for l in range(28):
        ls.append([])
    
    if toothies == 0:
        teethi = [1,2,3,4]
    elif toothies == 1:
        teethi = [5,6,7,8]
    else:
        teethi = [1,2,3,4,5,6,7,8]
        
    for i in teethi:
        lmi = read_all_landmarks_by_tooth(i)
        for k in range(len(lmi)):
            if len(ls[k]) == 0:
                ls[k] = lmi[k]
            else:
                ls[k] = np.concatenate((ls[k],lmi[k]),axis=0)
    return ls
    
    
def draw_landmark(img,lpts,color=(255,0,0),thicc=1):
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
    for lm in ls:
        draw_landmark(img,lm)
    
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
        new_mean = center_orig(np.reshape(new_mean,(lengthn,2))).flatten()
        # Normalize and align new mean to initial mean
        new_mean = normalize(align(new_mean,x0))
        # Re-center new mean
        new_mean = center_orig(np.reshape(new_mean,(lengthn,2))).flatten()

        # Check for convergence
        if (mean - new_mean < 0.000001).all():
            break
        else:
            mean = new_mean
    
    # Reshape the result
    ls = map(lambda e: np.reshape(e,(lengthn,2)),lsflat)
    mean = np.reshape(mean,(lengthn,2))
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

def convert_landmarklist_to_np(lms):
    # Leg alle vectoren plat
    xlms = map(lambda el: el.T,lms)
    # Maak een np.array, resultaat is 3D
    xlms = np.asarray(xlms,order='F')    
    # Strijk de vectoren plat in 2D
    xlms = xlms.reshape(28,2*lengthn)
    # shape is nu 28x80
    # print xlms.shape
    
    return xlms
    
def pca_reduce(lms,mean,num_comp):
    lms = convert_landmarklist_to_np(lms)
    eigenvals, eigenvecs, mean_n = pca(lms,num_comp)#,np.asarray(mean.T)
    mean_n = mean_n.reshape(1,320)
    print eigenvecs.shape
    terms = cv2.PCAProject(lms,mean_n,eigenvecs)
    
    return mean_n, eigenvecs, eigenvals, terms

def pca(X, nb_components=0):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    [n,d] = X.shape
    if (nb_components <= 0) or (nb_components>n):
        nb_components = n
    
    # Calculate the mean image and subtract it from each sample
    samples = X.copy()
    meanSample = samples.mean(axis=0)
    samplesMinMean = samples - meanSample 
    
    # Calculate the covariance matrix
    covMat = np.cov(samplesMinMean)
    
    # Get the eigen decomposition and order them
    valsRandom, vecsCovRandom = np.linalg.eig(covMat)
    
    sortedInd = valsRandom.argsort()[::-1]   
    vals = valsRandom[sortedInd]
    vecsCov = vecsCovRandom[:,sortedInd]
    
    # Normalize every eigenvector
    vecsImg = samplesMinMean.T.dot(vecsCov)
    lengths = np.linalg.norm(vecsImg,axis=0)
    vecs = vecsImg.T / lengths[:, np.newaxis]
    
    return vals, vecs[0:nb_components,:], meanSample
    
def pca_reconstruct(terms, mean, eigenvecs):
    
    return cv2.PCABackProject(terms, mean, eigenvecs)
    
    
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

def get_num_eigenvecs_needed(eigenvals):
    num = 0
    tot = sum(eigenvals)
    last = 0
    for i in range(len(eigenvals)):
        #num = num+1
        gain = sum(eigenvals[0:i+1])/tot - last
        new = last+gain
        print str(i) + ': ' + str(gain) + ' totalling ' + str(100*new) + '%'
        last = new
        #if sum(eigenvals[0:i+1])/tot > 0.95:
         #   return num
    return num
    
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
    # lm = read_all_landmarks_by_tooth(1)
    lm = read_all_landmarks_by_orientation(0)
    lm,mean = procrustes(lm)
    #draw_aligned_landmarks(img,lm)

    mean, eigenvecs, eigenvals, lm_reduced = pca_reduce(lm,mean,9)
    
    #maxNumVecsNeeded = get_num_eigenvecs_needed(eigenvals)
    stdvar = np.std(lm_reduced, axis=0)
    
    recon_lms = pca_reconstruct(lm_reduced, mean, eigenvecs)
    mean2 = mean.reshape(2,lengthn).T
    plus_one = pca_reconstruct(np.array([[-3*stdvar[0],0,0,0,0,0,0,0,0]]),mean, eigenvecs).reshape(2,lengthn).T
    #plus_one = pca_reconstruct(200*np.array([eigenvals[0:9]]),mean, eigenvecs).reshape(2,lengthn).T
    #plus_one = pca_reconstruct(np.array([[0,0,0,0,0,0,0,0,0]]),mean, eigenvecs).reshape(2,lengthn).T
    
    draw_aligned_landmarks(img,[mean2,plus_one])
    
    # draw_pca_reconstruction(img,recon_lms)
    cv2.imshow('image',img);
    # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # height, width, _ = img.shape;
    # cv2.resizeWindow('image', width / 2, height /2)

    
