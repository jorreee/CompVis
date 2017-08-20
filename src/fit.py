# -*- coding: utf-8 -*-
import cv2
import math
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
def find_new_points(imgtf, shapeo, edgeimgs, k, m):
    stop = False
    counter = 0
    shape = np.copy(shapeo)   
    normalvectors = gl.get_normals(shape)
    normalvectors = np.reshape(normalvectors,(normalvectors.size / 2, 2),'F')
    shape = np.reshape(shape,(shape.size / 2, 2),'F')
 
    #col = io.greyscale_to_colour(imgtf)
    #draw.draw_contour(col,shape)
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
            #dist[j] = np.dot(np.dot((own_gradient_profile[j:j+(2*k + 1)].flatten() - pmean).T,LA.pinv(pcov)),(own_gradient_profile[j:j+(2*k + 1)].flatten() - pmean))
            dist[j] = scp.mahalanobis(own_gradient_profile[j:j+(2*k + 1)].flatten(),pmean,LA.pinv(pcov))              
        #dist[k] = 0.85 * dist[k]
        min_ind = np.argmin(dist)
        lower = slices[0,:].size / 4.0 + slices[0,:].size / 8.0
        upper = lower + slices[0,:].size / 4.0
        #print '#####'
        #print lower
        #print upper
        #print min_ind
        #print str(min_ind + k)
        if (min_ind + k <= lower or min_ind + k >= upper):
            #print 'DING'
            counter += 1
        new_point = slices[:,min_ind+k]
        shape[i] = new_point
    
    #draw.draw_contour(col,shape,(0,0,255))
    #io.show_on_screen(col,1)
    
    print str(float(counter) / (shape.size/2))
    if float(counter) / (shape.size/2) < 0.05:
        stop = True
    shape = np.reshape(shape,(shape.size, 1),'F')  
    return shape, stop
    
   
def asm(imgtf, edgeimgs, b, tx, ty, s, theta, k, m, stdvar, mean, eigenvecs, maxiter):
    
    for i in range(maxiter):                
        shapetf = lm.pca_reconstruct(b,mean,eigenvecs)
        shapetf = lm.transform_shape(shapetf, tx, ty, s, theta)        
        approx, stop = find_new_points(imgtf, shapetf, edgeimgs, k, m)
        
        if stop:
            print i
            break
        b, tx, ty, s, theta = match_model_to_target(approx, mean, eigenvecs)
        
        #Check for plausible shapes
        for i in range(b.size):
            b[i, 0] = max(min(b[i, 0],3*stdvar[i]),-3*stdvar[i])
        
        if i == maxiter - 1:
            print i
    
    return b, tx, ty, s, theta

def srasm2(enhimgtf, edgeimgs, marks, orient, k, m, modes, maxiter):
    lms,_ = lm.procrustes(marks)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms, modes)
    stdvar = np.std(lm_reduced, axis=1)
    
    itx, ity, isc, itheta = init.get_initial_transformation(enhimgtf,mean,orient)
    b = np.zeros((modes,1))
    
    imgtf = pp.apply_sobel(enhimgtf)
    imgtf = cv2.pyrDown(imgtf)
    edgies = np.array(map(lambda x: cv2.pyrDown(x),edgeimgs))
    b, ntx, nty, nsc, itheta = asm(imgtf, edgies, b, itx / 2, ity / 2, isc / 2, itheta, k, m, stdvar, mean, eigenvecs, maxiter)      
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),ntx * 2,nty * 2,nsc * 2,itheta)

    
def srasm(enhimgtf, edgeimgs, marks, orient, k, m, modes, maxiter):
    lms,_ = lm.procrustes(marks)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms, modes)
    stdvar = np.std(lm_reduced, axis=1)
    
    itx, ity, isc, itheta = init.get_initial_transformation(enhimgtf,mean,orient)
    b = np.zeros((modes,1))
    
    imgtf = pp.apply_sobel(enhimgtf)
    b, ntx, nty, nsc, itheta = asm(imgtf, edgeimgs, b, itx, ity, isc, itheta, k, m, stdvar, mean, eigenvecs, maxiter)      
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),ntx,nty,nsc,itheta)

def mrasm(enhimgtf, edgeimgs, marks, orient, k, m, modes, maxiter):
    lms,_ = lm.procrustes(marks)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms, modes)
    stdvar = np.std(lm_reduced, axis=1)
    
    itx, ity, isc, itheta = init.get_initial_transformation(enhimgtf,mean,orient)
    b = np.zeros((modes,1))
    
    imgtf = pp.apply_sobel(enhimgtf)
    for i in range(2):
        edgies = edgeimgs
        limgtf =  imgtf
        j = 0
        while 1 - i > j:
            limgtf = cv2.pyrDown(limgtf)
            edgies = np.array(map(lambda x: cv2.pyrDown(x),edgies))
            j += 1 
        print edgies.shape
        b, ntx, nty, nsc, itheta = asm(limgtf, edgies, b, itx / math.pow(2.0,(1-i)), ity / math.pow(2.0,(1-i)), isc / math.pow(2.0,(1-i)), itheta, k, m, stdvar, mean, eigenvecs, maxiter)
        itx = ntx * math.pow(2.0,(1-i))
        ity = nty * math.pow(2.0,(1-i))
        isc = nsc * math.pow(2.0,(1-i))
        
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),itx,ity,isc,itheta)
    
def match_image(imgind, orientation = 2, showground = True, modes = 5, k = 5, m = 10, maxiter = 50):
    img = io.get_enhanced_img(imgind)
    imges = io.get_all_gradient_img(imgind)
    colgradimg = io.greyscale_to_colour(pp.apply_sobel(img))
    
    if orientation != 1:
        marks = io.read_all_landmarks_by_orientation(0,imgind)
        upper = srasm(img, imges, marks,0, k, m, modes, maxiter)
        draw.draw_contour(colgradimg,upper, thicc=1)
        
        if showground:
            allmarks = io.read_all_landmarks_by_orientation(0)
            upperground = np.reshape(allmarks[:,imgind - 1],(allmarks[:,imgind - 1].size,1))
            upperground = lm.transform_shape(upperground,-1150,-500,1,0)
            draw.draw_contour(colgradimg,upperground,color=(0,255,0), thicc=1)

        
    if orientation != 0:
        marks = io.read_all_landmarks_by_orientation(1,imgind)
        lower = srasm(img, imges, marks,1, k, m, modes, maxiter)
        draw.draw_contour(colgradimg,lower, thicc=1)
        
        if showground:
            allmarks = io.read_all_landmarks_by_orientation(1)
            lowerground = np.reshape(allmarks[:,imgind - 1],(allmarks[:,imgind - 1].size,1))
            lowerground = lm.transform_shape(lowerground,-1150,-500,1,0)
            draw.draw_contour(colgradimg,lowerground,color=(0,255,0), thicc=1)
    
          
    io.show_on_screen(colgradimg,1)
    return None
    
#def srasm4(enhimgtf, edgeimgs, marks, orient, k, m, modes, maxiter):
#    lms,_ = lm.procrustes(marks)
#    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms, modes)
#    stdvar = np.std(lm_reduced, axis=1)
#    
#    itx, ity, isc, itheta = init.get_initial_transformation(enhimgtf,mean,orient)
#    b = np.zeros((modes,1))
#    
#    imgtf = pp.apply_sobel(enhimgtf)
#    imgtf = cv2.pyrDown(cv2.pyrDown(imgtf))
#    edgies = np.array(map(lambda x: cv2.pyrDown(cv2.pyrDown(x)),edgeimgs))
#    b, ntx, nty, nsc, itheta = asm(imgtf, edgies, b, itx / 4, ity / 4, isc / 4, itheta, k, m, stdvar, mean, eigenvecs, maxiter)      
#    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),ntx * 4,nty * 4,nsc * 4,itheta)
    

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
        
        #img = io.get_objectspace_img()
        #draw.draw_aligned_contours(img,x)
    
        #5
        yprime = y / ( np.dot(y.flatten(),xbar.flatten()))
        
        #draw.draw_aligned_contours(img,xbar)
        #io.show_on_screen(img,1)
    
        #6
        lb = b
        b = np.dot(P.T,(yprime - xbar))
        if ((abs(b - lb) < conv_thresh).all() and abs(tx - ltx) < conv_thresh and 
            abs(ty - lty) < conv_thresh and abs(s - ls) < conv_thresh and abs(theta - ltheta) < conv_thresh):
            break;
    
    return b, tx, ty, s, theta   
    
if __name__ == '__main__':
    #match_image(12, orientation = 2, showground = True, modes = 5, k = 10, m = 20, maxiter = 0)
    
    imges = io.get_all_gradient_img(2)
    marks = io.read_all_landmarks_by_orientation(0,2)
    
    croppedmarks = np.array([])
    for i in range(13):
        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i],(marks[:,i].size,1)),-1150,-500,1,0))
    for i in range(13):
        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i + 13],(marks[:,i + 13].size,1)),-1173,-500,1,0))
    croppedmarks = np.reshape(croppedmarks,(croppedmarks.size / 26,26),'F') 
    
    wollah = io.greyscale_to_colour(imges[17])
    ground = np.reshape(croppedmarks[:,17],(croppedmarks[:,15].size,1))
    draw.draw_contour(wollah,ground,color=(0,0,255), thicc=1)
    io.show_on_screen(wollah,1)

    gl.get_statistical_model_new(imges,croppedmarks,5)
    
    #wollah = io.get_enhanced_img(2)
    #imges = io.get_all_gradient_img(2)
    #print imges.shape
    #marks = io.read_all_landmarks_by_orientation(0,2)
    #points = srasm2(wollah, imges, marks,0, 5, 5)
    #owollah = io.greyscale_to_colour(cv2.flip(pp.apply_sobel(wollah),1))
    #
    #marks2 = io.read_all_landmarks_by_orientation(0)
    #ground = np.reshape(marks2[:,14],(marks2[:,14].size,1))
    
    #!!!!!!!!!!!!!!!!
    # Landmark transformation for cutoff
    #ground = lm.transform_shape(ground,-1150,-500,1,0)
    # Mirrored landmark transformation for cutoff
    #ground = lm.transform_shape(ground,-1173,-500,1,0)
    
    
    
    #draw.draw_contour(owollah,ground,color=(0,0,255), thicc=1)
    #
    #draw.draw_contour(owollah,points, thicc=1)
    #io.show_on_screen(owollah,1)
    
    

    
