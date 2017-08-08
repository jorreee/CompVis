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
    #draw.draw_jaw_separation(smallerimg,uppery + sep-50)
    #draw.draw_jaw_separation(smallerimg,sep)
    #draw.draw_jaw_separation(smallerimg,lowery + sep)
    #colimgtf = io.greyscale_to_colour(smallerimg)
    #io.show_on_screen(colimgtf,1)
    upperyf = uppery + sep-50 + 250
    loweryf = lowery + sep + 250
    return upperyf, loweryf

# Gets the initial pose transformation to apply to the mean shape to start the ASM algorithm    
def get_initial_transformation(img,meanlm,orient):
    #meanlm = np.reshape(meanlm,(meanlm.size/2,2),'F')
    half = meanlm.size / 2
    upper, lower = get_jaw_separation(img)
    scalefactor = 900
    scale = scalefactor * np.sqrt(meanlm.size/80)
    rot = 0
    tx = 360
    #tx = 390
    if orient == 0:
        ty = upper + scale * np.min(meanlm[half:])
    elif orient == 1:
        ty = lower - scale * np.min(meanlm[half:])
    else:
        print "Only up and down are supported as of yet."
    ty = ty + 10
    return tx, ty, scale, rot
            
#def draw_initial_landmarks_orient(img,ls,orient):
#    sep = get_jaw_separation(img)
#    lss = np.copy(ls)
#    scalefactor = 900
#    scalen = scalefactor * np.sqrt(lm.lengthn/40)
#    if orient == 0:
#        ty = sep + (scalen * np.min(lss[0],0))[1]
#    elif orient == 1:
#        ty = sep - (scalen * np.min(lss[0],0))[1]
#    else:
#        print "Only up and down are supported as of yet."
#    tx = 360
#    for el in lss:
#        for i in range(0,len(el)):
#            el[i,0] = el[i,0] * scalen + tx
#            el[i,1] = el[i,1] * scalen + ty
#    for i in range(len(lss)):
#        if i == 0:
#            lm.draw_landmark(img,lss[i],color=(0,0,0),thicc=2)
#        else:
#            lm.draw_landmark(img,lss[i],color=(200,200,0),thicc=2)
#    
#    return lss[0]

#def draw_initial_landmarks(img,ls):
#    lss = ls
#    scalen = 900
#    tx = 360
#    ty = 310
#    for el in lss:
#        for i in range(0,len(el)):
#            el[i,0] = el[i,0] * scalen * np.sqrt(lm.lengthn/40) + tx
#            el[i,1] = el[i,1] * scalen * np.sqrt(lm.lengthn/40) + ty
#    for i in range(len(lss)):
#        if i == 0:
#            lm.draw_landmark(img,lss[i],color=(0,0,0),thicc=2)
#        else:
#            lm.draw_landmark(img,lss[i],color=(200,200,0),thicc=2)
#    return lss