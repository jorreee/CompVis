# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
import scipy.signal as scisig
import scipy.ndimage.morphology as morf
import landmarks as lm
import preprocess as pp
    
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
    
def draw_initial_landmarks_orient(img,ls,orient):
    sep = get_jaw_separation(img)
    lss = ls
    scalefactor = 900
    scalen = scalefactor * np.sqrt(lm.lengthn/40)
    if orient == 0:
        ty = sep + (scalen * np.min(mean,0))[1]
    elif orient == 1:
        ty = sep - (scalen * np.min(mean,0))[1]
    else:
        print "Only up and down are supported as of yet."
    tx = 360
    for el in lss:
        for i in range(0,len(el)):
            el[i,0] = el[i,0] * scalen + tx
            el[i,1] = el[i,1] * scalen + ty
    for i in range(len(lss)):
        if i == 0:
            lm.draw_landmark(img,lss[i],color=(0,0,0),thicc=2)
        else:
            lm.draw_landmark(img,lss[i],color=(200,200,0),thicc=2)
    return lss
    
def get_normals_single_tooth(pointlist):
    normals = np.zeros((40,2))
    for ind in range(40):
        prevp = ind-1
        nextp = ind+1
        if (ind == 0):
            prevp = 39
        elif (ind == 39):
            nextp = 0
        dx1 = pointlist[prevp,0] - pointlist[ind,0]
        dx2 = pointlist[ind,0] - pointlist[nextp,0]
        
        dy1 = pointlist[prevp,1] - pointlist[ind,1]
        dy2 = pointlist[ind,1] - pointlist[nextp,1]
        
        v1 = np.array([-dy1,dx1])
        v2 = np.array([-dy2,dx2])
        
        nv = lm.normalize(0.5*v1 + 0.5*v2)
        
        normals[ind,0] = nv[0]
        normals[ind,1] = nv[1]
        
    return normals
    
def get_normals(pointlist):
    normals = np.array([])
    for i in range(4):
        tnorms = get_normals_single_tooth(pointlist[i*40:(i+1)*40])
        if i == 0:
            normals = tnorms
        else:
            normals = np.append(normals,tnorms,axis=0)
        
    return normals
    
def draw_normals(img,lpts,norms):
    for ind in range(160):
        cv2.line(img,
                (int(float(lpts[ind,0])),int(float(lpts[ind,1]))),
                (   int(float(norms[ind,0] + lpts[ind,0])),
                    int(float(norms[ind,1] + lpts[ind,1]))),
                (255,0,0),
                2);
    
    return img
    
def get_img(i):
    if i < 10:
        img = cv2.imread("data/Radiographs/0"+str(i)+".tif")
    else:
        img = cv2.imread("data/Radiographs/"+str(i)+".tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[500:1430,1150:1850]
    #img = img[750:1150,1150:1850]
    
    return img
    
def get_all_edge_img(excepti):
    edges = np.array([])
    for i in range(13):
        if i == excepti:
            continue
        imgi = get_img(i+1)
        imgi = pp.apply_filter_train(imgi)
        imgi = pp.apply_sobel(imgi)
        np.append(edges,imgi)
    
    return edges
    
def get_single_slice_side(point,nv, k):    
    a = np.array(point)
    b = np.array(point + nv*k)
    coordinates = (a[:, np.newaxis] * np.linspace(1, 0, k+1) +
                   b[:, np.newaxis] * np.linspace(0, 1, k+1))
    coordinates = coordinates.astype(np.int)
    
    return coordinates
    
def get_slice(point,nv, k):    
    up = get_single_slice_side(point,nv,k)
    
    down = get_single_slice_side(point, -1*nv,k)
    down = np.delete(down,0,1)
    down = down[:,::-1]
    
    tot = np.concatenate((down,up),1)
    
    return tot
    
def fit(imgtf, mean, eigvs, edgeimgs):
    
    grad_imgi = pp.apply_filter_train(imgtf)
    grad_imgi = pp.apply_sobel(grad_imgi)
    
    normalvectors = get_normals(mean)
    normalvectors = normalvectors
    
    # Voor ieder punt
    for i in range(lm.lengthn):
        # Bereken de slice 
        slicel = get_slice(mean[i],normalvectors[i],5)
        own_gradient_profile = slice_image(slicel,grad_imgi)
        if i == 123:
            print mean[i]
            print normalvectors[i]
            print slicel
            print own_gradient_profile
    
    imgtf = draw_normals(imgtf,mean,normalvectors)
    
    return imgtf
    
def slice_image(coords,img):
    r, c = coords.shape
    values = np.ones(c)
    for i in range(c):
        values[i] = img[coords[0,i],coords[1,i]]
    
    return values
    
def get_jaw_separation(img):
    smallerimg = img[250:650,:]
    means = []
    for i in range(len(smallerimg[:,1])):
        means.append(np.mean(smallerimg[i,:]))
    print np.argmin(means)
    return np.argmin(means) + 250

def draw_jaw_separation(img,yval):
    for ind in range(len(img[1,:]) - 1):
        cv2.line(img,
                (ind,int(yval)),
                ((ind + 1),
                (int(yval))),
                (255,0,0),
                2);
    
    return None
    
    
    
if __name__ == '__main__':
    # Get image to fit
    img_to_fit = get_img(1)
    y = get_jaw_separation(img_to_fit)
    draw_jaw_separation(img_to_fit,y)
    
    # Get edge images to build statistical model from
    all_edge_imgs = get_all_edge_img(10)
    # Get shape we'll try to fit
    lms = lm.read_all_landmarks_by_orientation(1)
    # Build ASM
    lms,meano = lm.procrustes(lms)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms,meano,9)
    mean = mean.reshape(2,160).T
    draw_initial_landmarks_orient(img_to_fit,[mean],1) 
    #
    ##init = draw_initial_landmarks(img_to_fit,[mean])
    #
    ##img_to_fit = fit(img_to_fit, meano, eigenvecs, all_edge_imgs)
    #
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width = img_to_fit.shape;
    cv2.resizeWindow('image', width / 2, height /2)
    cv2.imshow('image',img_to_fit);

    
