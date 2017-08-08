# -*- coding: utf-8 -*-
import numpy as np


def get_jaw_separation(img):
    smallerimg = img[250:650,:]
    means = []
    for i in range(len(smallerimg[:,1])):
        means.append(np.mean(smallerimg[i,:]))
    
    return np.argmin(means) + 250

# Gets the initial pose transformation to apply to the mean shape to start the ASM algorithm    
def get_initial_transformation(img,meanlm,orient):
    #meanlm = np.reshape(meanlm,(meanlm.size/2,2),'F')
    half = meanlm.size / 2
    sep = get_jaw_separation(img)
    scalefactor = 900
    scale = scalefactor * np.sqrt(meanlm.size/80)
    rot = 0
    tx = 360
    if orient == 0:
        ty = sep + scale * np.min(meanlm[half:])
    elif orient == 1:
        ty = sep - scale * np.min(meanlm[half:])
    else:
        print "Only up and down are supported as of yet."
    return tx, ty, scale, rot