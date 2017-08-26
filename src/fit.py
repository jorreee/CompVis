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
import matplotlib.path as mppath

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
    #print str(float(counter) / (shape.size/4))
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
            print '  Stopped at iteration ' + str(i) + ' out of ' + str(maxiter) + ' maximum.'
            break
        b, tx, ty, s, theta = match_model_to_target(approx, mean, eigenvecs)
        
        #Check for plausible shapes
        for p in range(b.size):
            b[p, 0] = max(min(b[p, 0],3*stdvar[p]),-3*stdvar[p])
        
        if i == maxiter - 1:
            print '  Completed max number of iterations: ' + str(i + 1)
            
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


def mrasm(enhimgtf, edgeimgs, marks, orient, k, m, modes, maxiter, resdepth = 1):
###### TIJDELIJK VOOR REPORT
    # imgr = io.greyscale_to_colour(io.get_img(1))
    # draw.draw_square(imgr,np.array([[0, 700],[0, 930]]))
    # for i in range(26):
    #     shapei = draw.move_into_frame(np.copy(marks[:,i]))
    #     draw.draw_contour(imgr,shapei,color=(150 - 4*i,222,20 + 9*i),thicc=2)
    # 
    # io.show_on_screen(imgr,1)
    # if True:
    #     return None
###### EINDE TIJDELIJK
    lms,_ = lm.procrustes(marks)
###### TIJDELIJK VOOR REPORT
    # imgr = io.greyscale_to_colour(io.get_img(1))
    # draw.draw_square(imgr,np.array([[0, 700],[0, 930]]))
    # for i in range(26):
    #     shapei = draw.make_object_space_visible(np.copy(lms[:,i]))
    #     draw.draw_contour(imgr,shapei,color=(150 - 4*i,222,20 + 9*i),thicc=2)
    # 
    # io.show_on_screen(imgr,1)
    # if True:
    #     return None
###### EINDE TIJDELIJK
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(np.copy(lms), 26)
    coverage, modes_needed = lm.get_num_eigenvecs_needed(eigenvals)
    # draw.vec2graph(coverage)
    # draw.vec2graph(0.95*np.ones(26))
    # print modes_needed
    mean, eigenvecs, eigenvals, lm_reduced = lm.pca_reduce(lms, modes_needed)
    stdvar = np.std(lm_reduced, axis=1)
###### TIJDELIJK VOOR REPORT
    # imgr = io.greyscale_to_colour(io.get_img(1))
    # for i in range(5):
    #     draw.draw_square(imgr,np.array([[0, 700],[0, 930]]))
    #     for k in range(3):
    #         termsplus = np.zeros((5,1))
    #         termsmin = np.zeros((5,1))
    #         termsplus[i] = (k+1)*stdvar[i]
    #         termsmin[i] = -1*(k+1)*stdvar[i]
    #         draw.draw_contour(imgr,draw.make_object_space_visible(lm.pca_reconstruct(termsplus,mean,eigenvecs)),color=(10,63 + k*63,100),thicc=2)
    #         draw.draw_contour(imgr,draw.make_object_space_visible(lm.pca_reconstruct(termsmin,mean,eigenvecs)),color=(10,100,63 + k*63),thicc=2)
    #     
    #     draw.draw_contour(imgr,draw.make_object_space_visible(mean),color=(0,230,230),thicc=3)
    #     io.show_on_screen(imgr,1)
###### EINDE TIJDELIJK
    
    croppedmarks = np.array([])
    for i in range(13):
        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i],(marks[:,i].size,1)),-1150,-500,1,0))
    for i in range(13):
        croppedmarks = np.append(croppedmarks,lm.transform_shape(np.reshape(marks[:,i + 13],(marks[:,i + 13].size,1)),-1173,-500,1,0))
    croppedmarks = np.reshape(croppedmarks,(croppedmarks.size / 26,26),'F') 
    
    itx, ity, isc, itheta = init.get_initial_transformation(enhimgtf,mean,orient)
    b = np.zeros((modes_needed,1))
    
    print ''
    print 'Starting multi-resolution fitting procedure for orientation ' + str(orient)
    print '--------'
    imgtf = pp.apply_sobel(enhimgtf)
    for i in range(resdepth + 1):
        print 'Step ' + str(i) + ':'
        edgies = edgeimgs
        limgtf =  imgtf
        j = 0
        while resdepth - i > j:
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
        b, ntx, nty, nsc, itheta = asm(limgtf, means, covs, b, float(itx) / math.pow(2.0,(resdepth-i)), float(ity) / math.pow(2.0,(resdepth-i)), float(isc) / math.pow(2.0,(resdepth-i)), itheta, k, m, stdvar, mean, eigenvecs, maxiter, orient)
        itx = ntx * math.pow(2.0,(resdepth-i))
        ity = nty * math.pow(2.0,(resdepth-i))
        isc = nsc * math.pow(2.0,(resdepth-i))
        
    return lm.transform_shape(lm.pca_reconstruct(b,mean,eigenvecs),itx,ity,isc,itheta)
    
def match_image(imgind, orientation = 2, showground = True, modes = 5, k = 5, m = 10, maxiter = 50, multires = True):
    img = io.get_enhanced_img(imgind)
    imges = io.get_all_gradient_img(imgind)
    colgradimg = io.greyscale_to_colour(io.get_img(imgind))
    #colgradimg = io.greyscale_to_colour(pp.apply_sobel(img))
    
    # Init evaluation
    upper = None
    upperground = None
    lower = None
    lowerground = None
    
    if orientation != 1:
        marks = io.read_all_landmarks_by_orientation(0,imgind)
        if multires:
            upper = mrasm(img, imges, marks,0, k, m, modes, maxiter)
        else:
            upper = srasm2(img, imges, marks,0, k, m, modes, maxiter)
        draw.draw_contour(colgradimg,upper, thicc=1)
        
        allmarks = io.read_all_landmarks_by_orientation(0)
        upperground = None
        upperground = np.reshape(allmarks[:,imgind - 1],(allmarks[:,imgind - 1].size,1))
        upperground = lm.transform_shape(upperground,-1150,-500,1,0)
            
        if showground:
            draw.draw_contour(colgradimg,upperground,color=(0,255,0), thicc=1)

    if orientation != 0:
        marks = io.read_all_landmarks_by_orientation(1,imgind)
        if multires:
            lower = mrasm(img, imges, marks,1, k, m, modes, maxiter)
        else:
            lower = srasm2(img, imges, marks,1, k, m, modes, maxiter)            
        draw.draw_contour(colgradimg,lower, thicc=1)
        
        allmarks = io.read_all_landmarks_by_orientation(1)
        lowerground = None
        lowerground = np.reshape(allmarks[:,imgind - 1],(allmarks[:,imgind - 1].size,1))
        lowerground = lm.transform_shape(lowerground,-1150,-500,1,0)
        if showground:
            draw.draw_contour(colgradimg,lowerground,color=(0,255,0), thicc=1)
    
    #io.show_on_screen(colgradimg,1)
    #cv2.imwrite("result_contour.png",colgradimg)
    
    dumpimg = io.greyscale_to_colour(io.get_img(imgind))
    resultvec = evaluate_results(upper, upperground, lower, lowerground, True, dumpimg)
    cv2.imwrite(str(imgind) + "i" + str(maxiter) + '.png',dumpimg)
    #io.show_on_screen(dumpimg,1)
    return resultvec
    
def evaluate_results(upo, uprefo, lowo, lowrefo, showResults=False, img=None):
    evalvec = np.ones((1,7))
    numpts = upo.size / 2
    
    up = np.reshape(np.copy(upo),(numpts, 2),'F')
    upref = np.reshape(np.copy(uprefo),(numpts, 2),'F')
    low = np.reshape(np.copy(lowo),(numpts, 2),'F')
    lowref = np.reshape(np.copy(lowrefo),(numpts, 2),'F')
    
    print ''
    print '--- Evaluation ---'
    
    #Euclidische afstand
    uptot = 0
    lowtot = 0
    for i in range(numpts):
        uptot += math.hypot(up[i,0] - upref[i,0], up[i,1] - upref[i,1])
        lowtot += math.hypot(low[i,0] - lowref[i,0], low[i,1] - lowref[i,1])
    print '    1) Average Euclidean distance to ground truth: ' + str((uptot + lowtot) / numpts) 
    evalvec[0,0] = (uptot + lowtot) / numpts
    print '                                                   ('+ str(uptot*2.0 / numpts) +' upper,' 
    evalvec[0,1] = uptot*2.0 / numpts
    print '                                                    '+ str(lowtot*2.0 / numpts) +' lower)'
    evalvec[0,2] = lowtot*2.0 / numpts

    
    # Contained points 
    numTP = 0
    numFP = 0
    numFN = 0
    inCount = 0
    inrefCount = 0
    for i in range(4):
        # UPPER
        #
        #Bereken de bounding box van ground en gevonden mark
        toothiup = np.concatenate((up[(40*i):(40*(i+1)),:],upref[(40*i):(40*(i+1)),:]))
        bbup = bbox(toothiup)
        #Maak een Path object van de ground en gevonden mark
        upp = mppath.Path(up[(40*i):(40*(i+1)),:])      
        uprefp = mppath.Path(upref[(40*i):(40*(i+1)),:])
        #Loop door alle pixels in de bounding box
        for r in range(int(bbup[1,0]),int(bbup[1,1])):
            for c in range(int(bbup[0,0]),int(bbup[0,1])):
                #Klassificeer
                if uprefp.contains_point(np.array([c,r])):
                    inrefCount += 1
                    if upp.contains_point(np.array([c,r])):
                        inCount += 1
                        numTP += 1
                        if showResults:
                            draw.draw_pixel(img,np.array([c,r]),(10,180,10))
                    else:
                        numFN += 1
                        if showResults:
                            draw.draw_pixel(img,np.array([c,r]),(10,180,180))
                elif upp.contains_point(np.array([c,r])):
                    inCount += 1
                    numFP += 1
                    if showResults:
                        draw.draw_pixel(img,np.array([c,r]),(10,10,180))
                    
        # LOWER
        #
        #Bereken de bounding box van ground en gevonden mark
        toothilow = np.concatenate((low[(40*i):(40*(i+1)),:],lowref[(40*i):(40*(i+1)),:]))
        bblow = bbox(toothilow)
        #Maak een Path object van de ground en gevonden mark
        lowp = mppath.Path(low[(40*i):(40*(i+1)),:])      
        lowrefp = mppath.Path(lowref[(40*i):(40*(i+1)),:])
        #Loop door alle pixels in de bounding box
        for r in range(int(bblow[1,0]),int(bblow[1,1])):
            for c in range(int(bblow[0,0]),int(bblow[0,1])):
                #Klassificeer
                if lowrefp.contains_point(np.array([c,r])):
                    inrefCount += 1
                    if lowp.contains_point(np.array([c,r])):
                        inCount += 1
                        numTP += 1
                        if showResults:
                            draw.draw_pixel(img,np.array([c,r]),(10,180,10))
                    else:
                        numFN += 1
                        if showResults:
                            draw.draw_pixel(img,np.array([c,r]),(10,180,180))
                elif lowp.contains_point(np.array([c,r])):
                    inCount += 1
                    numFP += 1
                    if showResults:
                        draw.draw_pixel(img,np.array([c,r]),(10,10,180))
    
    print '    2) Surface analysis: ' + str(numTP / float(inrefCount)) + ' percent of ground truth surface found'
    evalvec[0,3] = numTP / float(inrefCount)
    print '                         ' + str(numFN / float(inrefCount)) + ' percent of ground truth surface missed'
    evalvec[0,4] = numFN / float(inrefCount)
    print '                         ' + str(numTP / float(inCount)) + ' percent of solution surface within ground'
    evalvec[0,5] = numTP / float(inCount)
    print '                         ' + str(numFP / float(inCount)) + ' percent of solution surface outside ground'
    evalvec[0,6] = numFP / float(inCount)
    
    return evalvec
    
# Geïnspireerd op https://stackoverflow.com/questions/12443688/calculating-bounding-box-of-numpy-array
def bbox(points):
    """
    [xmin xmax]
    [ymin ymax]
    """
    a = np.zeros((2,2))
    a[:,0] = np.min(points, axis=0)
    a[:,1] = np.max(points, axis=0)
    
    a[0,0] = int(math.floor(a[0,0]))
    a[1,0] = int(math.floor(a[1,0]))
    a[0,1] = int(math.ceil(a[0,1]))
    a[1,1] = int(math.ceil(a[1,1]))
    
    return a
    
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
    print ''
    print 'So... it has come to this.'
    print ''
##### REPORT
    # evals = np.ones((14*3,7))
    # numiter = [10, 20, 50]
    # 
    # for i in range(14):
    #     for ji in range(3):
    #         j = numiter[ji]
    #         print ''
    #         print '##### ' + str(i+1) + 'i' + str(j)
    #         evals[(i*3)+ji,:] = match_image(i+1, orientation = 2, showground = True, modes = 5, k = 5, m = 10, maxiter = j, multires = True)
    # 
    # print ''
    # print 'Eval matrix: [Eucl tot, Eucl up, Eucl low, TP / ground, FN / ground, TP / sol, FP / sol]'
    # print evals
##### EINDE REPORT
    
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
    
    

    
