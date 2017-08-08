# -*- coding: utf-8 -*-
import numpy as np
import draw as draw
import io as io
import greylevel as gl


def get_jaw_separation(img):
    smallerimg = img[250:650,150:550]
    means = []
    for i in range(len(smallerimg[:,1])):
        means.append(np.mean(smallerimg[i,:]))
    
    sep = np.argmin(means) #+ 250
    
    upper = means[sep-50:sep]
    lower = means[sep:sep+65]
    gradupper = gl.get_gradients(np.array(upper))
    gradlower = gl.get_gradients(np.array(lower))
    uppery = np.argmax(gradupper)
    lowery = np.argmax(gradlower)
    upperyf = uppery + sep-50 + 250
    loweryf = lowery + sep + 250
    return upperyf, loweryf

def get_centralisation(img, yvalo):
    yval = yvalo - 250
    start = 100
    eind = 601
    smallerimg = img[250:650,start:eind]
    
    reepje = smallerimg[yval,:]
    reepgrad = gl.get_gradients_raw(np.asarray(reepje))
    #draw.vec2graph(reepgrad)
    
    # Geef voorrang aan pieken dicht bij het centrum
    mid = (eind-start) / 2
    for i in range(mid):
        reepgrad[i] *= (float(i)/mid)
        reepgrad[mid + i] *= (float(mid-i)/mid)
    #draw.vec2graph(reepgrad)
    
    x_offset = np.argmax(reepgrad) - mid
    print x_offset
    return x_offset

# Gets the initial pose transformation to apply to the mean shape to start the ASM algorithm    
def get_initial_transformation(img,meanlm,orient):
    #meanlm = np.reshape(meanlm,(meanlm.size/2,2),'F')
    half = meanlm.size / 2
    upper, lower = get_jaw_separation(img)
    if orient == 0:
        scalefactor = 900
        scale = scalefactor * np.sqrt(meanlm.size/80)
        ty = upper + scale * np.min(meanlm[half:]) + 10
        tx = 350 + get_centralisation(img,upper)
        rot = 0
    elif orient == 1:
        scalefactor = 750
        scale = scalefactor * np.sqrt(meanlm.size/80)
        ty = lower - scale * np.min(meanlm[half:]) - 10
        tx = 350 + get_centralisation(img,lower)
        rot = 0.08
    else:
        print "Only up and down are supported as of yet."
    tx = 360
    return tx, ty, scale, rot
