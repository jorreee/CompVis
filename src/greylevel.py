# -*- coding: utf-8 -*-
import numpy as np
import landmarks as lm
   
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
    

def get_single_slice_side(point, nv, k):    
    a = np.array(point)
    b = np.array(point + nv*k)
    coordinates = (a[:, np.newaxis] * np.linspace(1, 0, k+1) +
                   b[:, np.newaxis] * np.linspace(0, 1, k+1))
    coordinates = coordinates.astype(np.int)
    
    return coordinates
    
def get_slice_indices(point,nv, k):    
    up = get_single_slice_side(point,nv,k)
    
    down = get_single_slice_side(point, -1*nv,k)
    down = np.delete(down,0,1)
    down = down[:,::-1]
    
    tot = np.concatenate((down,up),1)
    
    return tot
    
def slice_image(coords,img):
    r, c = coords.shape
    values = np.ones(c)
    for i in range(c):
        values[i] = img[coords[0,i],coords[1,i]]
    
    return lm.normalize(values)
    
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
    