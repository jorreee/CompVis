# -*- coding: utf-8 -*-
import cv2
import numpy as np
import landmarks as lm; reload(lm)
import preprocess as pp
import scipy.spatial.distance as scp
import greylevel as gl; reload(gl)
import io as io; reload(io)
import init as init; reload(init)
import draw as draw; reload(draw)
import sys as sys
from numpy import linalg as LA

# Finds the new landmarks points by using the grey level model of a certain shape
def find_new_points(imgtf, shapeo, edgeimgs, k):
    shape = np.copy(shapeo)
    m = 2*k    
    normalvectors = gl.get_normals(shape)
    normalvectors = np.reshape(normalvectors,(normalvectors.size / 2, 2),'F')
    shape = np.reshape(shape,(shape.size / 2, 2),'F')
    # Voor ieder punt
    for i in range(shape.size / 2):
        # Bereken stat model
        slicel = gl.get_slice_indices(shape[i],normalvectors[i],k)
        pmean, pcov = gl.get_statistical_model(edgeimgs,slicel,k)
        
        # Bereken eigen sample
        slices = gl.get_slice_indices(shape[i],normalvectors[i],m)
        own_gradient_profile = gl.slice_image(slices,imgtf)
        
        dist = np.zeros(2*(m - k) + 1)
        for j in range(0,2*(m - k) + 1):
            # Pinv gives the same result as inv for Mahalanobis
            dist[j] = scp.mahalanobis(own_gradient_profile[j:j+(2*k + 1)],pmean,LA.pinv(pcov))
            #if is_singular(pcov):
            #    dist[j] = scp.mahalanobis(own_gradient_profile[j:j+(2*k + 1)],pmean,LA.pinv(pcov))
            #else:
            #    dist[j] = scp.mahalanobis(own_gradient_profile[j:j+(2*k + 1)],pmean,LA.inv(pcov))               
        min_ind = np.argmin(dist)
        new_point = slices[:,min_ind+k]
        shape[i] = new_point
    shape = np.reshape(shape,(shape.size, 1),'F')  
    return shape
    
def fit(imgtf, mean, eigvs, edgeimgs, k, orient):
    itx, ity, isc, itheta = init.get_initial_transformation(imgtf,mean,orient)
    shape = lm.transform_shape(mean,itx,ity,isc,itheta)
    for i in range(50):
        approx = find_new_points(imgtf, shape, edgeimgs, k)
        b, tx, ty, s, theta = match_model_to_target(approx, mean, eigenvecs)
        shape = lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),tx,ty,s,theta)
    
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),tx,ty,s,theta)

# Does not work robustly?     
def is_singular(matrix):
    if LA.cond(matrix) < 1/sys.float_info.epsilon:
        return False
    else:
        return True

# Finds the shape parameter terms b and the pose transformation T(tx,ty,s,theta) 
#  that best matches the model x = xbar + P * b (in object space) tot the new points Y (in image space).
# The parameters P represents the eigenvectors   
def match_model_to_target(Y, xbar, P):
    conv_thresh = 0.000001 
    #1. 
    b = np.reshape(np.zeros(P[0].size),(P[0].size,1))
    lb = np.reshape(np.zeros(P[0].size),(P[0].size,1))
    tx = ty = theta = 0
    s = 1
    ltx = lty = ltheta = 0
    ls = 1
    
    while True:
        #2.
        x = lm.pca_reconstruct(b, xbar, P)
    
        #3.
        ltx = tx
        lty = ty
        ls = s
        ltheta = theta
        tx, ty, s, theta = lm.get_align_params(x, Y)
    
        #4.
        y = lm.transform_shape_inv(Y,tx, ty, s, theta)
    
        #5
        yprime = y / ( np.dot(y.flatten(),xbar.flatten()))
    
        #6
        lb = b
        b = np.dot(P.T,(yprime - xbar)) 
        if ((b - lb < conv_thresh).all() and tx - ltx < conv_thresh and 
            ty - lty < conv_thresh and s - ls < conv_thresh and theta - ltheta < conv_thresh):
            break;
    
    return b, tx, ty, s, theta   
    
if __name__ == '__main__':
    owollah = io.get_img(1)
    wollah = io.get_enhanced_img(1)
    imges = io.get_all_enhanced_img(1)
    marks = io.read_all_landmarks_by_orientation(0)
    lms,_ = lm.procrustes(marks)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms,9)
    points = fit(wollah, mean, eigenvecs, imges, 2, 0)
    owollah = io.greyscale_to_colour(owollah)
    tx, ty, s, r = init.get_initial_transformation(wollah,mean,0)
    transformedmean = lm.transform_shape(mean,tx,ty,s,r)
    draw.draw_contour(owollah,points, thicc=2)
    io.show_on_screen(owollah,2)
    
    #draw.draw_contour(wollah,transformedmean)
    #io.show_on_screen_greyscale(wollah,2)
    #lss = draw_initial_landmarks_orient(img_to_fit,[mean],0)
    
    ##singlemark = np.reshape(marks[:,0],(marks[:,0].size,1))
    #normies = gl.get_normals(transformedmean)
    #half = transformedmean.size /2
    ###print (normies[0,0],normies[half,0])
    #sliceu = gl.get_slice_indices(np.array([transformedmean[0,0],transformedmean[half,0]]),np.array([normies[0,0],normies[half,0]]),5)
    #pmean, pcov = gl.get_statistical_model(imges,sliceu,5)
    ##all_edge_imgs = io.get_all_enhanced_img(1)
    
    

    # # Get image to fit
    # img_to_fit = get_img(1)
    # 
    # y = get_jaw_separation(img_to_fit)
    # draw_jaw_separation(img_to_fit,y)
    # 
    # # Get edge images to build statistical model from
    # all_edge_imgs = get_all_edge_img(1)
    # 
    # # Get shape we'll try to fit
    # lms = lm.read_all_landmarks_by_orientation(0)
    # # Build ASM
    # lms,meano = lm.procrustes(lms)
    # mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms,meano,9)
    # mean = mean.reshape(2,160).T
    # lss = draw_initial_landmarks_orient(img_to_fit,[mean],0)
    # 
    # #init = draw_initial_landmarks(img_to_fit,[mean])
    # 
    # #img_to_fit = 
    # newlmraw = fit(img_to_fit, lss, eigenvecs, all_edge_imgs, 5)
    # 
    # newlm = match_model_to_target(newlmraw, lss, eigenvecs)
    # 
    # lm.draw_landmark(img_to_fit,newlmraw)
    # 
    # #draw_initial_landmarks_orient(img_to_fit,[mean],0) 
    # 
    # #init = draw_initial_landmarks(img_to_fit,[mean])

    
