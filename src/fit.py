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
from numpy import linalg as LA

# Finds the new landmarks points by using the grey level model of a certain shape
def find_new_points(imgtf, shapeo, means, covs, k, m, orient):
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
        #slicel = gl.get_slice_indices(shape[i],normalvectors[i],k)
        #pmean, pcov = gl.get_statistical_model(edgeimgs,slicel,k)
        
        # Bereken eigen sample
        slices = gl.get_slice_indices(shape[i],normalvectors[i],m)
        own_gradient_profile = gl.slice_image(slices,imgtf)
        
        dist = np.zeros(2*(m - k) + 1)
        for j in range(0,2*(m - k) + 1):
            # Pinv gives the same result as inv for Mahalanobis
            #dist[j] = np.dot(np.dot((own_gradient_profile[j:j+(2*k + 1)].flatten() - pmean).T,LA.pinv(pcov)),(own_gradient_profile[j:j+(2*k + 1)].flatten() - pmean))
            dist[j] = scp.mahalanobis(own_gradient_profile[j:j+(2*k + 1)].flatten(),means[i],LA.pinv(covs[i]))              
        min_ind = np.argmin(dist)
        lower = slices[0,:].size / 4.0 #+ slices[0,:].size / 8.0
        upper = lower + slices[0,:].size / 2.0
        #print '#####'
        #print lower
        #print upper
        #print min_ind
        #print str(min_ind + k)
        if (orient == 0 and i % 40 >= 10 and i % 40 < 30 and  (min_ind + k < lower or min_ind + k > upper)):
            counter += 1
        elif (orient == 1 and (i % 40 < 8 or i % 40 >= 28) and  (min_ind + k < lower or min_ind + k > upper)):
            counter += 1
        new_point = slices[:,min_ind+k]
        shape[i] = new_point
    #draw.draw_contour(col,shape,(0,0,255))
    #io.show_on_screen(col,1)
    print str(float(counter) / (shape.size/4))
    if float(counter) / (shape.size/4) < 0.1:
        stop = True
    shape = np.reshape(shape,(shape.size, 1),'F')  
    return shape, stop
    
   
def asm(imgtf, means, covs, b, tx, ty, s, theta, k, m, stdvar, mean, eigenvecs, maxiter, orient):
    for i in range(maxiter):                 
        shapetf = lm.pca_reconstruct(b,mean,eigenvecs)
        shapetf = lm.transform_shape(shapetf, tx, ty, s, theta)   
        approx, stop = find_new_points(imgtf, shapetf, means, covs, k, m, orient)
        
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

def srasm(enhimgtf, edgeimgs, marks, orient, k, m, modes, maxiter):
    lms,_ = lm.procrustes(marks)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms, modes)
    stdvar = np.std(lm_reduced, axis=1)
    
    croppedmarks = np.array([])
    for i in range(13):
        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i],(marks[:,i].size,1)),-1150,-500,1,0))
    for i in range(13):
        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i + 13],(marks[:,i + 13].size,1)),-1173,-500,1,0))
    croppedmarks = np.reshape(croppedmarks,(croppedmarks.size / 26,26),'F') 
    means, covs = gl.get_statistical_model_new(edgeimgs,croppedmarks,k)
    
    itx, ity, isc, itheta = init.get_initial_transformation(enhimgtf,mean,orient)
    b = np.zeros((modes,1))
    
    imgtf = pp.apply_sobel(enhimgtf)
    b, ntx, nty, nsc, itheta = asm(imgtf, means, covs, b, itx, ity, isc, itheta, k, m, stdvar, mean, eigenvecs, maxiter, orient)      
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),ntx,nty,nsc,itheta)
    
    
def srasm2(enhimgtf, edgeimgs, marks, orient, k, m, modes, maxiter):
    lms,_ = lm.procrustes(marks)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms, modes)
    stdvar = np.std(lm_reduced, axis=1)
    
    croppedmarks = np.array([])
    for i in range(13):
        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i],(marks[:,i].size,1)),-1150,-500,1,0))
    for i in range(13):
        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i + 13],(marks[:,i + 13].size,1)),-1173,-500,1,0))
    croppedmarks = np.reshape(croppedmarks,(croppedmarks.size / 26,26),'F') 
    smallermarks = np.array([])
    for i in range(26):
        smallermarks = np.append(smallermarks,lm.transform_shape(np.reshape(croppedmarks[:,i],(croppedmarks[:,i].size,1)),0,0,1.0 / 2,0))
    smallermarks = np.reshape(smallermarks,(smallermarks.size / 26,26),'F') 
    
    itx, ity, isc, itheta = init.get_initial_transformation(enhimgtf,mean,orient)
    b = np.zeros((modes,1))
    
    imgtf = pp.apply_sobel(enhimgtf)
    imgtf = cv2.pyrDown(imgtf)
    edgies = np.array(map(lambda x: cv2.pyrDown(x),edgeimgs))
    means, covs = gl.get_statistical_model_new(edgies,smallermarks,k)
    b, ntx, nty, nsc, itheta = asm(imgtf, means, covs, b, itx / 2, ity / 2, isc / 2, itheta, k, m, stdvar, mean, eigenvecs, maxiter, orient)      
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),ntx * 2,nty * 2,nsc * 2,itheta)


def mrasm(enhimgtf, edgeimgs, marks, orient, k, m, modes, maxiter):
    lms,_ = lm.procrustes(marks)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms, modes)
    stdvar = np.std(lm_reduced, axis=1)
    
    croppedmarks = np.array([])
    for i in range(13):
        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i],(marks[:,i].size,1)),-1150,-500,1,0))
    for i in range(13):
        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i + 13],(marks[:,i + 13].size,1)),-1173,-500,1,0))
    croppedmarks = np.reshape(croppedmarks,(croppedmarks.size / 26,26),'F') 
    
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
            smallermarks = np.array([])
            for l in range(26):
                smallermarks = np.append(smallermarks,lm.transform_shape(np.reshape(croppedmarks[:,l],(croppedmarks[:,l].size,1)),0,0,1.0 / 2,0))
            smallermarks = np.reshape(smallermarks,(smallermarks.size / 26,26),'F') 
            j += 1 
        
        #wollah = io.greyscale_to_colour(edgies[18])
        #ground = np.reshape(smallermarks[:,18],(smallermarks[:,12].size,1))
        #draw.draw_contour(wollah,ground,color=(0,0,255), thicc=1)
        #io.show_on_screen(wollah,1)
        means, covs = gl.get_statistical_model_new(edgies,smallermarks,k)
        b, ntx, nty, nsc, itheta = asm(limgtf, means, covs, b, float(itx) / math.pow(2.0,(1-i)), float(ity) / math.pow(2.0,(1-i)), float(isc) / math.pow(2.0,(1-i)), itheta, k, m, stdvar, mean, eigenvecs, maxiter, orient)
        itx = ntx * math.pow(2.0,(1-i))
        ity = nty * math.pow(2.0,(1-i))
        isc = nsc * math.pow(2.0,(1-i))
        
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),itx,ity,isc,itheta)
    
def match_image(imgind, orientation = 2, showground = True, modes = 5, k = 5, m = 10, maxiter = 50, multires = True):
    img = io.get_enhanced_img(imgind)
    imges = io.get_all_gradient_img(imgind)
    colgradimg = io.greyscale_to_colour(io.get_img(imgind))
    #colgradimg = io.greyscale_to_colour(pp.apply_sobel(img))
    
    if orientation != 1:
        marks = io.read_all_landmarks_by_orientation(0,imgind)
        if multires:
            upper = mrasm(img, imges, marks,0, k, m, modes, maxiter)
        else:
            upper = srasm2(img, imges, marks,0, k, m, modes, maxiter)
        draw.draw_contour(colgradimg,upper, thicc=1)
        
        if showground:
            allmarks = io.read_all_landmarks_by_orientation(0)
            upperground = None
            upperground = np.reshape(allmarks[:,imgind - 1],(allmarks[:,imgind - 1].size,1))
            upperground = np.reshape(allmarks[:,imgind - 1],(allmarks[:,imgind - 1].size,1))
            upperground = lm.transform_shape(upperground,-1150,-500,1,0)
            draw.draw_contour(colgradimg,upperground,color=(0,255,0), thicc=1)

        
    if orientation != 0:
        marks = io.read_all_landmarks_by_orientation(1,imgind)
        if multires:
            lower = mrasm(img, imges, marks,1, k, m, modes, maxiter)
        else:
            lower = srasm2(img, imges, marks,1, k, m, modes, maxiter)            
        draw.draw_contour(colgradimg,lower, thicc=1)
        
        if showground:
            allmarks = io.read_all_landmarks_by_orientation(1)
            lowerground = None
            lowerground = np.reshape(allmarks[:,imgind - 1],(allmarks[:,imgind - 1].size,1))
            lowerground = lm.transform_shape(lowerground,-1150,-500,1,0)
            draw.draw_contour(colgradimg,lowerground,color=(0,255,0), thicc=1)
    
          
    io.show_on_screen(colgradimg,1)
    cv2.imwrite("result.png",colgradimg)
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
    match_image(1, orientation = 2, showground = True, modes = 5, k = 5, m = 10, maxiter = 10, multires = True)
    
    #img = cv2.flip(io.get_enhanced_img(1),1)
#    img = io.get_enhanced_img(1)
#    img = pp.apply_sobel(img)
#    imges = io.get_all_gradient_img(1)
#    marks = io.read_all_landmarks_by_orientation(1,1)
##    
#    croppedmarks = np.array([])
#    for i in range(13):
#        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i],(marks[:,i].size,1)),-1150,-500,1,0))
#    for i in range(13):
#        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i + 13],(marks[:,i + 13].size,1)),-1173,-500,1,0))
#    croppedmarks = np.reshape(croppedmarks,(croppedmarks.size / 26,26),'F') 
#    
#    wollah = io.greyscale_to_colour(cv2.pyrDown(imges[2]))
#    ground = np.reshape(croppedmarks[:,2],(croppedmarks[:,12].size,1))
#    half = ground.size / 2
#    second = np.append(ground[40:48,0],ground[half+40:half+48,0])
#    first = np.append(ground[68:80,0],ground[half+68:half+80,0])
#    #ground = lm.transform_shape(ground,- big[:,0].size / 2, - big[0,:].size / 2,1 / 2,0)
#    
#    ground2 = lm.transform_shape(second,0, 0,1.0 / 2,0)
#    ground = lm.transform_shape(first,0, 0,1.0 / 2,0)
#    draw.draw_contour(wollah,ground,color=(0,0,255), thicc=1)
#    draw.draw_contour(wollah,ground2,color=(0,0,255), thicc=1)
#    io.show_on_screen(wollah,1)

    #gl.get_statistical_model_new(imges,croppedmarks,5)
    
    
    
    
    
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
    
    

    
