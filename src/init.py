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
    get_centralisation(img,uppery)
    #draw.draw_jaw_separation(smallerimg,uppery + sep-50)
    #draw.draw_jaw_separation(smallerimg,sep)
    #draw.draw_jaw_separation(smallerimg,lowery + sep)
    #colimgtf = io.greyscale_to_colour(smallerimg)
    #io.show_on_screen(colimgtf,1)
    upperyf = uppery + sep-50 + 250
    loweryf = lowery + sep + 250
    return upperyf, loweryf

def get_centralisation(img, yval):
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
    
    return x_offset

# Gets the initial pose transformation to apply to the mean shape to start the ASM algorithm    
def get_initial_transformation(img,meanlm,orient):
    #meanlm = np.reshape(meanlm,(meanlm.size/2,2),'F')
    half = meanlm.size / 2
    upper, lower = get_jaw_separation(img)
    scalefactor = 900
    scale = scalefactor * np.sqrt(meanlm.size/80)
    rot = 0
    tx = 324
    #tx = 360
    if orient == 0:
        ty = upper + scale * np.min(meanlm[half:])
    elif orient == 1:
        ty = lower - scale * np.min(meanlm[half:])
    else:
        print "Only up and down are supported as of yet."
    ty = ty + 10
    return tx, ty, scale, rot
