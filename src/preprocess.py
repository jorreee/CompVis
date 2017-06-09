# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch


def create_database(directory, show = True):
    '''
    Process all images in the given directory.
    Every image is cropped to the detected face, resized to 100x100 and save in another directory (orignal directory name + "2").
    
    @param directory:    directory to process
    @param show:         bool, show all intermediate results
    
    '''
    #load a pre-trained classifier
    cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt.xml")
    #loop through all files
    for filename in fnmatch.filter(os.listdir(directory),'*.jpg'):
        file_in = directory+"/"+filename
        file_out= directory+"2/"+filename
        img = cv2.imread(file_in)
        if show:
            cv2.imshow('img',img)
            cv2.waitKey(0)
        #do face detection    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
        rects[:,2:] += rects[:,:2]
        rects = rects[0,:]
        #show detected result
        if show:
            vis = img.copy()
            cv2.rectangle(vis, (rects[0], rects[1]), (rects[2], rects[3]), (0, 255, 0), 2)
            cv2.imshow('img', vis)
            cv2.waitKey(0)
        #crop image to the rectangle and resample it to 100x100 pixels 
        
        result = cv2.resize(img[rects[1]:rects[3], rects[0]:rects[2]],(100,100))
        
        #show result
        if show:
            cv2.imshow('img', result)
            cv2.waitKey(0)            
        #save the face in a second directory
        cv2.imwrite(file_out, result)
    cv2.destroyAllWindows()



def createX(directory,nbDim=10000):
    '''
    Create an array that contains all the images in directory.
    @return np.array, shape=(nb images in directory, nb pixels in image)
    '''
    filenames = fnmatch.filter(os.listdir(directory),'*.jpg')
    nbImages = len(filenames)
    X = np.zeros( (nbImages,nbDim) )#, dtype=np.uint8 )
    for i,filename in enumerate( filenames ):
        file_in = directory+"/"+filename
        img = cv2.imread(file_in)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        X[i,:] = gray.flatten()
    print X.dtype
    return X


def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''
    return np.dot(np.transpose(W) , X - mu.flat)

def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    return np.dot(W,Y) + mu.flat

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

    
    #calculate scrambled covariance matrix for increased performance
    [covar, mean] = cv2.calcCovarMatrix(X, cv.CV_COVAR_SCALE |  cv.CV_COVAR_ROWS | cv.CV_COVAR_SCRAMBLED)
    #compute eigenvalues and vectors of scrambled covariance matrix
    [retval,eigenvals,eigenvects] = cv2.eigen(covar, True)
    #calculate the normal eigenvactors from the scrambled eigenvectors
    eigenvects = cv2.gemm(eigenvects, X - mean, 1, None, 0)
    eigenvects = np.apply_along_axis(lambda n: cv2.normalize(n).flat, 1, eigenvects)
    # return the nb_components largest eigenvalues and eigenvectors
    return [eigenvals[:n], np.transpose(eigenvects[:n]), np.transpose(mean)]

def normalize(img):
    '''
    Normalize an image such that it min=0 , max=255 and type is np.uint8
    '''
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)

if __name__ == '__main2__':
    #create database of normalized images
    for directory in ["data/arnold", "data/barack"]:
        create_database(directory, show = False)
    
    show = True
    
    #create big X arrays for arnold and barack
    Xa = createX("data/arnold2")
    Xb = createX("data/barack2")
            
    #Take one part of the images for the training set, the rest for testing
    nbTrain = 6
    Xtest = np.vstack( (Xa[nbTrain:,:],Xb[nbTrain:,:]) )
    Ctest = ["arnold"]*(Xa.shape[0]-nbTrain) + ["barack"]*(Xb.shape[0]-nbTrain)
    Xa = Xa[0:nbTrain,:]
    Xb = Xb[0:nbTrain,:]

    #do pca
    [eigenvaluesa, eigenvectorsa, mua] = pca(Xa,nb_components=6)
    [eigenvaluesb, eigenvectorsb, mub] = pca(Xb,nb_components=6)
    #visualize
    cv2.imshow('img',np.hstack( (mua.reshape(100,100),
                                 normalize(eigenvectorsa[:,0].reshape(100,100)),
                                 normalize(eigenvectorsa[:,1].reshape(100,100)),
                                 normalize(eigenvectorsa[:,2].reshape(100,100)))
                               ).astype(np.uint8))
    cv2.waitKey(0) 
    cv2.imshow('img',np.hstack( (mub.reshape(100,100),
                                 normalize(eigenvectorsb[:,0].reshape(100,100)),
                                 normalize(eigenvectorsb[:,1].reshape(100,100)),
                                 normalize(eigenvectorsb[:,2].reshape(100,100)))
                               ).astype(np.uint8))
    cv2.waitKey(0) 
            
    nbCorrect = 0
    for i in range(Xtest.shape[0]):
        X = Xtest[i,:]
        
        #project image i on the subspace of arnold and barack
        Ya = project(eigenvectorsa, X, mua )
        Xa= reconstruct(eigenvectorsa, Ya, mua)
        
        Yb = project(eigenvectorsb, X, mub )
        Xb= reconstruct(eigenvectorsb, Yb, mub)
        if show:
            #show reconstructed images
            cv2.imshow('img',np.hstack( (X.reshape(100,100),
                                         np.clip(Xa.reshape(100,100), 0, 255),
                                         np.clip(Xb.reshape(100,100), 0, 255)) ).astype(np.uint8) )
            cv2.waitKey(0)   

        #classify i
        if np.linalg.norm(Xa-Xtest[i,:]) < np.linalg.norm(Xb-Xtest[i,:]):
            bestC = "arnold"
        else:
            bestC = "barack"
        print str(i)+":"+str(bestC)
        if bestC == Ctest[i]:
            nbCorrect+=1
    
    #Print final result
    print str(nbCorrect)+"/"+str(len(Ctest))
    
    
if __name__ == '__main__':
    img = cv2.imread("data/Radiographs/01.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    tmp = open('data/Landmarks/original/landmarks1-1.txt').readlines();
    for ind in range(0, len(tmp), 2):
        if (ind + 2 >= len(tmp)):
            cv2.line(img,(int(float(tmp[ind].strip())),int(float(tmp[ind +1].strip()))),(int(float(tmp[0].strip())),int(float(tmp[1].strip()))),(255,0,0),2);
        else :
            cv2.line(img,(int(float(tmp[ind].strip())),int(float(tmp[ind +1].strip()))),(int(float(tmp[ind +2].strip())),int(float(tmp[ind +3].strip()))),(255,0,0),2);
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width, _ = img.shape;
    cv2.resizeWindow('image', width / 2, height /2)
    cv2.imshow('image',img);

    
