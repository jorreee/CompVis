# -*- coding: utf-8 -*-
import numpy as np
import landmarks as lm

# Returns the normals n = (nx0, nx1, ..., ny0, ny1, ...)^T of a list of landmarks x = (x0, x1, ..., y0, y1, ...)^T  
def get_normals(pointlist):
    half = pointlist.size/2
    pointlist = np.reshape(pointlist,(half,2),'F')
    normals = np.zeros((half,2))
    for ind in range(half):
        prevp = ind-1
        nextp = ind+1
        if (ind % 40 == 0):
            prevp = ind + 39
        elif (ind % 40 == 39):
            nextp = ind - 39
        dx1 = pointlist[prevp,0] - pointlist[ind,0]
        dx2 = pointlist[ind,0] - pointlist[nextp,0]
        
        dy1 = pointlist[prevp,1] - pointlist[ind,1]
        dy2 = pointlist[ind,1] - pointlist[nextp,1]
        
        v1 = np.array([-dy1,dx1])
        v2 = np.array([-dy2,dx2])
        
        nv = lm.normalize_shape(0.5*v1 + 0.5*v2)
        
        normals[ind,0] = nv[0]
        normals[ind,1] = nv[1]
    
    normals = np.reshape(normals,(normals.size, 1),'F')    
    return normals
    
# Gets the slice indices starting from the point along k pixels in the direction of nv. 
# Point and nv are represented as (x,y).
def get_single_slice_side(point, nv, k):    
    a = np.array(point)
    b = np.array(point + nv*k)
    coordinates = (a[:, np.newaxis] * np.linspace(1, 0, k+1) +
                   b[:, np.newaxis] * np.linspace(0, 1, k+1))
    coordinates = coordinates.astype(np.int)
    
    return coordinates

# Gets the slice indices in both directions starting from the point along k pixels in the direction of -nv and k +1 pixels in the direction of nv (for gradients, DEPRECATED) 
# Point and nv are represented as (x,y).
# Returns an array with the shape of (2,2(k + 1)), due to the gradient needing one extra pixel  
def get_slice_indices(point,nv, k): 
    up = get_single_slice_side(point,nv,k)
    # (Deprecated) One extra in the up direction for gradient   
    #up = get_single_slice_side(point,nv,k + 1)
    
    down = get_single_slice_side(point, -1*nv,k)
    down = np.delete(down,0,1)
    down = down[:,::-1]
    
    tot = np.concatenate((down,up),1)
    
    return tot

# Gets the gradient of the grey values of an image along the slice provided by the 2k + 1 coordinates
# Returns 2k + 1 gradients
def slice_image(coords,img):
    r, c = coords.shape
    values = np.ones(c)
    for i in range(c):
        values[i] = img[coords[1,i],coords[0,i]]
    #gradients = get_gradients(values)
    gradients = values
    return lm.normalize_shape(gradients)
    
# Returns the gradients of the grey values along a slice provided by the 2(k + 1) greyvals
# Returns 2k + 1 gradients
def get_gradients(greyvals):
    values = np.ones(greyvals.size - 1)
    for i in range(values.size):
        values[i] = abs(greyvals[i + 1] - greyvals[i])
    return values
    
# Returns the raw (negatives, too) gradients of the grey values along a slice provided by the 2(k + 1) greyvals
# Returns 2k + 1 gradients
def get_gradients_raw(greyvals):
    values = np.ones(greyvals.size - 1)
    for i in range(values.size):
        values[i] = abs(int(greyvals[i + 1]) - int(greyvals[i]))
    return values

# Gets the grey level statistical model for the slice provided by the coordinates    
def get_statistical_model(imgs,coords,k):
    gradvals = np.array([])
    numel = len(imgs)
    for i in range(numel):
        if len(gradvals) == 0:
            gradvals = np.asarray(slice_image(coords,imgs[i]))
        else:
            gradvals = np.append(gradvals,slice_image(coords,imgs[i]))
    gradvals = gradvals.reshape(numel,2*k+1)
    mean = np.mean(gradvals,0)
    cov = np.cov(gradvals.T)
       
    return mean, cov
    
# Gets the grey level model of the 26 imgs (26*930*700) with the 26 marks (320*26) (mirrored included, explaining the 26).
def get_statistical_model_new(imgs,marks,k):
    half = marks[:,0].size / 2
    #  Get the normals in the same shape as the marks (320*26)
    normals = np.array([])
    for i in range(26):
        normals = np.append(normals,get_normals(np.reshape(marks[:,i - 1],(marks[:,i - 1].size,1))))
    normals = np.reshape(normals,(normals.size / 26,26),'F')  
    for i in range(half):
        gradvals = np.array([])
        for j in range(26):
            slicecoords = get_slice_indices(np.array([marks[i,j],marks[i+half,j]]),np.array([normals[i,j],normals[i+half,j]]),k)
            if len(gradvals) == 0:
                gradvals = np.asarray(slice_image(slicecoords,imgs[j]))
            else:
                gradvals = np.append(gradvals,slice_image(slicecoords,imgs[j]))
        gradvals = gradvals.reshape(26,2*k+1)
        mean = np.mean(gradvals,0)
        cov = np.cov(gradvals.T)
    
    #Alle mean en covs moeten nog op dit niveau opgevangen worden en teruggegeven worden

        