# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy import linalg as LA
import math


lengthn = 160

#Transform a shape x with a scale factor s, rotation factor theta and translation t
def transform_shape(xo,tx=0, ty=0, s=1, theta=0):
    x = np.copy(xo)
    x = np.reshape(x,(x.size / 2, 2),'F').flatten()
    srm = np.array([[s * np.cos(theta), -s * np.sin(theta)], [s * np.sin(theta), s * np.cos(theta)]])
    for k in range(0,len(x),2):
        nr = np.dot(srm,[x[k],x[k+1]])
        x[k] = nr[0] + tx
        x[k+1] = nr[1] + ty
    x = np.reshape(x,(x.size / 2, 2))
    x = np.reshape(x,(x.size, 1),'F')
    return x
    
#Inverse transform a shape x with a scale factor s, rotation factor theta and translation t
def transform_shape_inv(xo,tx=0, ty=0, s=1, theta=0):
    x = np.copy(xo)
    x = np.reshape(x,(x.size / 2, 2),'F').flatten()
    srm = np.array([[s * np.cos(theta), -s * np.sin(theta)], [s * np.sin(theta), s * np.cos(theta)]])
    srm = LA.inv(srm)
    for k in range(0,len(x),2):
        x[k] = x[k] - tx
        x[k+1] = x[k+1] - ty
        nr = np.dot(srm,[x[k],x[k+1]])
        x[k] = nr[0]
        x[k+1] = nr[1]
    x = np.reshape(x,(x.size / 2, 2))
    x = np.reshape(x,(x.size, 1),'F')
    return x
    

# Aligns a list of shapes on the first element of that list. 
# Returns the collection of aligned shapes and the mean shape.
#TODO
def procrustes(ls):
    
    #Translate landmarks to origin
    ls = center_orig_all(ls)
    
    #Choose first example for initial mean estimate and rescale
    x0 = ls[:,0]
    x0 = normalize_shape(x0)
    mean = x0

    while True:
        # Align all landmarks with mean
        ls = ls.T
        for i in range(0,len(ls[:,0])) :
            ls[i] = align(ls[i], mean).flatten()
        ls = ls.T
        # Re-estimate the mean
        new_mean = estimate_mean(ls)
        # Center new mean
        new_mean = center_orig(new_mean)
        # Normalize and align new mean to initial mean
        new_mean = normalize_shape(align(new_mean,x0))
        # Re-center new mean
        new_mean = center_orig(new_mean)

        # Check for convergence
        if (mean - new_mean < 0.000001).all():
            break
        else:
            mean = new_mean
             
    return [ls,mean]
    
# Extracts the centroid of a shape
def extract_centroid(lpts):
    # Get the dividing index between x and y coords
    half = lpts.size / 2
    # Get the flattened mean coordinates [x,y]
    centroidflat = np.array([np.mean(lpts[:half]),np.mean(lpts[half:])])
    # Reshape the mean coordinates to [x,y]^T
    return np.reshape(centroidflat,(2,1))

# Center one landmark to origin
def center_orig(mark):
    #Extract the centroid
    centroid = extract_centroid(mark)
    mark = mark.flatten()
    half = mark.size / 2
    # Subtract the centroid from the landmark
    result = np.concatenate((mark[:half] - centroid[0,0],mark[half:] - centroid[1,0]),axis=0)
    # Reshape the result to a single column
    return np.reshape(result,(mark.size,1))
    
    
# Center all shapes ls to origin
# A shape is represented by a column
def center_orig_all(ls):
    def flatten_center(mark):
        return center_orig(mark).flatten()
    # Center all the columns to the origin
    return np.apply_along_axis(flatten_center,0,ls)
    
def normalize_shape(v):
    norm=LA.norm(v)
    if norm==0: 
       return v
    return np.reshape(v/norm,(v.size,1))

# Estimates the mean shape of a set of shapes
def estimate_mean(ls):
    ss = np.sum(ls,1)
    return np.reshape(ss/(len(ls[0,:])),(ss.size,1))
    

#Aligns shape x1 to x2
def align(x1o,x2o):
    x1 = np.reshape(x1o,(x1o.size / 2, 2),'F').flatten()
    x2 = np.reshape(x2o,(x2o.size / 2, 2),'F').flatten()
    x1norm2 = np.dot(x1,x1)
    a = np.dot(x1,x2) / x1norm2
    b = 0
    for j in range(0,len(x1),2):
        b += x1[j] * x2[j+1] - x1[j+1] * x2[j]
    b /= x1norm2
    s = math.sqrt(a * a + b * b)
    theta = np.arctan(b/a)
    alignedx1 = transform_shape(x1o,0,0,s,theta)
    return alignedx1
   
#Gets the alignment parameters to align shape x1 to x2
def get_align_params(x1,x2):
    x1c = extract_centroid(x1)
    x2c = extract_centroid(x2)
    
    t = x2c - x1c
    
    x1 = center_orig(x1)
    x2 = center_orig(x2)
    x1 = np.reshape(x1,(x1.size / 2, 2),'F').flatten()
    x2 = np.reshape(x2,(x2.size / 2, 2),'F').flatten()   
    
    x1norm2 = np.dot(x1,x1)
    a = np.dot(x1,x2) / x1norm2
    b = 0
    for j in range(0,len(x1),2):
        b += x1[j] * x2[j+1] - x1[j+1] * x2[j]
    b /= x1norm2
    s = math.sqrt(a * a + b * b)
    theta = np.arctan(b/a)
        
    return t[0,0],t[1,0], s, theta

#TODO deprecated
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

# Extracts a PCA model from the given list of shapes, using the specified number
# of components. 
#
# Returns     the mean shape as a column vector, 
#             a matrix with the num_comp largest eigenvectors as columns, 
#             a column vector containing ALL eigenvalues, sorted large to small,
#             a matrix containing the PCA terms for all input shapes as columns
#
def pca_reduce(lmls,num_comp):
    #lms = convert_landmarklist_to_np(lms)
    #TODO vorm
    lms = np.copy(lmls).T
    eigenvals, eigenvecs, mean_n = pca(lms,num_comp)#,np.asarray(mean.T)
    mean_n = np.reshape(mean_n,(mean_n.shape[0],1))
    eigenvals = np.reshape(eigenvals,(eigenvals.shape[0],1))
    
    terms = cv2.PCAProject(lms.T,mean_n,eigenvecs)
    
    return mean_n, eigenvecs.T, eigenvals, terms

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
    
# Input: terms = matrix containing terms as columns
#        mean = mean as column vector
#        eigenvecs = matrix containing eigenvectors as columns
def pca_reconstruct(terms, mean, eigenvecs):
    return cv2.PCABackProject(terms, mean, eigenvecs.T)

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
    
#if __name__ == '__main__':
    #marks = io.read_all_landmarks_by_tooth(0)
    #img = io.get_objectspace_img()
    #[lm,mn] = procrustes(marks)
    #draw.draw_aligned_contours(img,lm)
    #io.show_on_screen(img)
    
    #normies = gl.get_normals(np.reshape(marks[:,0],(marks[:,0].size,1)))
    # Lege witte image maken voor procrustes afbeelding
    #img = np.zeros((512,512,3), np.uint8)
    #img[:] = (255, 255, 255)
    #img = cv2.imread("data/Radiographs/01.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    
    
    #draw.draw_all_contours(img, np.reshape(marks[:,0],(marks[:,0].size,1)))
    #draw.draw_normals(img, np.reshape(marks[:,0],(marks[:,0].size,1)), normies)
    #[lm,mn] = procrustes(marks)
    #mean, eigenvecs, eigenvals, lm_reduced = pca_reduce(lm,9)
    
    #stdvar = np.std(lm_reduced, axis=1)
    #recon_lms = pca_reconstruct(lm_reduced, mean, eigenvecs)
    #mean2 = mean.reshape(2,lengthn).T
    #plus_one = pca_reconstruct(np.array([[3*stdvar[0],0,0,0,0,0,0,0,0]]).flatten(),mean, eigenvecs)
    #plus_one = pca_reconstruct(200*np.array([eigenvals[0:9]]),mean, eigenvecs).reshape(2,lengthn).T
    #plus_one = pca_reconstruct(np.array([[0,0,0,0,0,0,0,0,0]]),mean, eigenvecs).reshape(2,lengthn).T
    #draw.draw_aligned_contours(img,plus_one)
    #draw.draw_aligned_contours(img,plus_one)
    
    
    # Dental radiograph inladen
    #img = cv2.imread("data/Radiographs/01.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    
    # Lege witte image maken voor procrustes afbeelding
    #img = np.zeros((512,512,3), np.uint8)
    #img[:] = (255, 255, 255)
    
    #tmp = open('data/Landmarks/original/landmarks1-1.txt').readlines();
    #for ind in range(0, len(tmp), 2):
    #    if (ind + 2 >= len(tmp)):
    #        cv2.line(img,(int(float(tmp[ind].strip())),int(float(tmp[ind +1].strip()))),(int(float(tmp[0].strip())),int(float(tmp[1].strip()))),(255,0,0),2);
    #    else :
    #        cv2.line(img,(int(float(tmp[ind].strip())),int(float(tmp[ind +1].strip()))),(int(float(tmp[ind +2].strip())),int(float(tmp[ind +3].strip()))),(255,0,0),2);
    # lm = read_all_landmarks_by_tooth(1)
    #lm = read_all_landmarks_by_orientation(0)
    #lm,mean = procrustes(lm)
    #draw_aligned_landmarks(img,lm)

    #mean, eigenvecs, eigenvals, lm_reduced = pca_reduce(lm,mean,9)
    
    #maxNumVecsNeeded = get_num_eigenvecs_needed(eigenvals)
    #stdvar = np.std(lm_reduced, axis=0)
    
    #recon_lms = pca_reconstruct(lm_reduced, mean, eigenvecs)
    #mean2 = mean.reshape(2,lengthn).T
    #plus_one = pca_reconstruct(np.array([[-3*stdvar[0],0,0,0,0,0,0,0,0]]),mean, eigenvecs).reshape(2,lengthn).T
    #plus_one = pca_reconstruct(200*np.array([eigenvals[0:9]]),mean, eigenvecs).reshape(2,lengthn).T
    #plus_one = pca_reconstruct(np.array([[0,0,0,0,0,0,0,0,0]]),mean, eigenvecs).reshape(2,lengthn).T
    
    #draw_aligned_landmarks(img,[mean2,plus_one])
    
    # draw_pca_reconstruction(img,recon_lms)
    #cv2.imshow('image',img);
    # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # height, width, _ = img.shape;
    # cv2.resizeWindow('image', width / 2, height /2)