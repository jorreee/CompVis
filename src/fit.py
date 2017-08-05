# -*- coding: utf-8 -*-
import cv2
import numpy as np
import landmarks as lm
import preprocess as pp
import scipy.spatial.distance as scp
import greylevel as gl; reload(gl)
import io as io; reload(io)
import init as init; reload(init)
import draw as draw; reload(draw)
    
def fit(imgtf, mean, eigvs, edgeimgs, k):
    #print mean.shape
    m = 2*k
    enh_imgtf = pp.apply_filter_train(imgtf)
    #grad_imgi = pp.apply_sobel(grad_imgi)
    
    normalvectors = gl.get_normals(mean)
    normalvectors = normalvectors
    
    approx = np.copy(mean)
    # Voor ieder punt
    for i in range(lm.lengthn):
        # Bereken stat model
        slicel = gl.get_slice_indices(approx[i],normalvectors[i],k)
        pmean, pcov = gl.get_statistical_model(edgeimgs,slicel,k)
        
        # Bereken eigen sample
        slices = gl.get_slice_indices(approx[i],normalvectors[i],m)
        own_gradient_profile = gl.slice_image(slices,enh_imgtf)
        
        dist = np.zeros(2*(m - k) + 1)
        for j in range(0,2*(m - k) + 1):
            dist[j] = scp.mahalanobis(own_gradient_profile[j:j+(2*k + 1)],pmean,np.linalg.pinv(pcov))
        min_ind = np.argmin(dist)
        new_point = slices[:,min_ind+k]
        #print str(approx[i]) + " becomes " + str(new_point)
        approx[i] = new_point
      
    # imgtf = draw_normals(imgtf,mean,normalvectors)
    
    return approx
    

    
def match_model_to_target(newpoints, model, eigenvecs):
    #1. 
    b = np.zeros(9)
    lastb = np.zeros(9)
    
    #2.
    x = cv2.PCABackProject(b,model.reshape(320,1),eigenvecs)
    
    #3.
    t, s, theta = lm.get_align_params(x, newpoints)
    
    #4.
    
    
    return None    
    
if __name__ == '__main__':
    wollah = io.get_img(1)
    imges = io.get_all_enhanced_img(1)
    marks = io.read_all_landmarks_by_orientation(0)
    lms,_ = lm.procrustes(marks)
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms,9)
#    mean = mean.reshape(2,160).T
    tx, ty, s, r = init.get_initial_transformation(wollah,mean,0)
    transformedmean = lm.transform_shape(mean,tx,ty,s,r)
    draw.draw_contour(wollah,transformedmean)
    io.show_on_screen_greyscale(wollah,2)
    #lss = draw_initial_landmarks_orient(img_to_fit,[mean],0)
    
    #singlemark = np.reshape(marks[:,0],(marks[:,0].size,1))
    normies = gl.get_normals(transformedmean)
    half = transformedmean.size /2
    ##print (normies[0,0],normies[half,0])
    sliceu = gl.get_slice_indices(np.array([transformedmean[0,0],transformedmean[half,0]]),np.array([normies[0,0],normies[half,0]]),5)
    pmean, pcov = gl.get_statistical_model(imges,sliceu,5)
    #all_edge_imgs = io.get_all_enhanced_img(1)
    
    

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

    
