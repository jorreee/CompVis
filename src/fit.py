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
    


# Imgtf is enhanced image, edgeimgs are gradient images    
def asm(imgtf, edgeimgs, b, tx, ty, s, theta, k, stdvar, mean, eigenvecs):
    
    for i in range(200):        
        #print i
        col = io.greyscale_to_colour(imgtf)
        
        shapetf = lm.pca_reconstruct(b,mean,eigenvecs)
        shapetf = lm.transform_shape(shapetf, tx, ty, s, theta)        
        approx, stop = find_new_points(imgtf, shapetf, edgeimgs, k, 2*k)
    
        draw.draw_contour(col,shapetf,(0,255,0))
        #draw.draw_contour(col,approx)
        
        if stop:
            print i
            break
        b, tx, ty, s, theta = match_model_to_target(approx, mean, eigenvecs)
        
        #Check for plausible shapes
        for i in range(b.size):
            b[i, 0] = max(min(b[i, 0],3*stdvar[i]),-3*stdvar[i])
            
        newshape = lm.pca_reconstruct(b,mean,eigenvecs)
        newshape = lm.transform_shape(newshape, tx, ty, s, theta)
        draw.draw_contour(col,newshape,(0,0,255))
        if i == 199:
            io.show_on_screen(col,1)
    
    return b, tx, ty, s, theta
    #result = lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),tx,ty,s,theta)

    #ground = np.reshape(marks[:,12],(marks[:,12].size,1))
    #ground = lm.transform_shape(ground,-1150,-500,1,0)
    #draw.draw_contour(colimgtf,ground,color=(0,0,255), thicc=1)

    #draw.draw_contour(colimgtf,first,color=(0,255,0), thicc=1)
    #draw.draw_contour(colimgtf,approx,color=(0,0,255), thicc=1)
    #draw.draw_contour(colimgtf,result, thicc=1)
    #io.show_on_screen(colimgtf,1)
    
    #return result

def srasm2(enhimgtf, edgeimgs, marks, orient, k, modes):
    lms,_ = lm.procrustes(marks)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms, modes)
    stdvar = np.std(lm_reduced, axis=1)
    
    itx, ity, isc, itheta = init.get_initial_transformation(enhimgtf,mean,orient)
    b = np.zeros((modes,1))
    
    imgtf = pp.apply_sobel(enhimgtf)
    imgtf = cv2.pyrDown(imgtf)
    edgies = np.array(map(lambda x: cv2.pyrDown(x),edgeimgs))
    b, ntx, nty, nsc, itheta = asm(imgtf, edgies, b, itx / 2, ity / 2, isc / 2, itheta, k, stdvar, mean, eigenvecs)      
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),ntx * 2,nty * 2,nsc * 2,itheta)

def srasm4(enhimgtf, edgeimgs, marks, orient, k, modes):
    lms,_ = lm.procrustes(marks)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms, modes)
    stdvar = np.std(lm_reduced, axis=1)
    
    itx, ity, isc, itheta = init.get_initial_transformation(enhimgtf,mean,orient)
    b = np.zeros((modes,1))
    
    imgtf = pp.apply_sobel(enhimgtf)
    imgtf = cv2.pyrDown(cv2.pyrDown(imgtf))
    edgies = np.array(map(lambda x: cv2.pyrDown(cv2.pyrDown(x)),edgeimgs))
    b, ntx, nty, nsc, itheta = asm(imgtf, edgies, b, itx / 4, ity / 4, isc / 4, itheta, k, stdvar, mean, eigenvecs)      
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),ntx * 4,nty * 4,nsc * 4,itheta)
    
def srasm(enhimgtf, edgeimgs, marks, orient, k, modes):
    lms,_ = lm.procrustes(marks)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms, modes)
    stdvar = np.std(lm_reduced, axis=1)
    
    itx, ity, isc, itheta = init.get_initial_transformation(enhimgtf,mean,orient)
    b = np.zeros((modes,1))
    
    imgtf = pp.apply_sobel(enhimgtf)
    b, ntx, nty, nsc, itheta = asm(imgtf, edgeimgs, b, itx, ity, isc, itheta, k, stdvar, mean, eigenvecs)      
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),ntx,nty,nsc,itheta)

def mrasm(enhimgtf, edgeimgs, marks, orient, k, modes):
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
        b, ntx, nty, nsc, itheta = asm(limgtf, edgies, b, itx / math.pow(2.0,(1-i)), ity / math.pow(2.0,(1-i)), isc / math.pow(2.0,(1-i)), itheta, k, stdvar, mean, eigenvecs)
        itx = ntx * math.pow(2.0,(1-i))
        ity = nty * math.pow(2.0,(1-i))
        isc = nsc * math.pow(2.0,(1-i))
        
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),itx,ity,isc,itheta)
    
    #colimgtf = io.greyscale_to_colour(imgtf) 
    #draw.draw_contour(colimgtf,first,color=(0,255,0), thicc=1)
    #io.show_on_screen(colimgtf,1)
    #
    #smalleritx = itx /2
    #smallerity = ity /2
    #smallerisc = isc /2
    #smallershape = lm.transform_shape(mean,smalleritx,smallerity,smallerisc,itheta)
    #smallercolimgtf = cv2.pyrDown(colimgtf)
    #draw.draw_contour(smallercolimgtf,smallershape,color=(0,255,0), thicc=1)
    #io.show_on_screen(smallercolimgtf,1)

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
    wollah = io.get_enhanced_img(4)
    imges = io.get_all_gradient_img(4)
    marks = io.read_all_landmarks_by_orientation(0,4)
    points = srasm2(wollah, imges, marks,0, 5, 5)
    owollah = io.greyscale_to_colour(pp.apply_sobel(wollah))
    
    marks2 = io.read_all_landmarks_by_orientation(0)
    ground = np.reshape(marks2[:,3],(marks2[:,3].size,1))
    ground = lm.transform_shape(ground,-1150,-500,1,0)
    draw.draw_contour(owollah,ground,color=(0,0,255), thicc=1)
    
    draw.draw_contour(owollah,points, thicc=1)
    io.show_on_screen(owollah,1)
    
    
    
    #owollah = io.greyscale_to_colour(owollah)
    #tx, ty, s, r = init.get_initial_transformation(wollah,mean,0)
    #transformedmean = lm.transform_shape(mean,tx,ty,s,r)
    #draw.draw_contour(owollah,points, thicc=2)
    #io.show_on_screen(owollah,2)
    
    
    
    
    
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

    
