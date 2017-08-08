# -*- coding: utf-8 -*-
import numpy as np
import draw as draw
import io as io
import greylevel as gl


def get_jaw_separation(img):
    smallerimg = img[250:650,150:550]
    smallerimgx = img[250:650,:]
    means = []
    for i in range(len(smallerimg[:,1])):
        means.append(np.mean(smallerimg[i,:]))
    
    sep = np.argmin(means) #+ 250
    
    upper = means[sep-100:sep]
    lower = means[sep:sep+100]
    gradupper = gl.get_gradients(np.array(upper))
    gradlower = gl.get_gradients(np.array(lower))
    uppery = np.argmax(gradupper)
    rapeje = smallerimgx[uppery,:]
    draw.vec2graph(rapeje)
    print rapeje
    rapegrad = gl.get_gradients_raw(np.asarray(rapeje))
    print '--------------'
    print rapegrad
    draw.vec2graph(rapegrad)
    lowery = np.argmax(gradlower)
    draw.draw_jaw_separation(smallerimg,uppery + sep-100)
    draw.draw_jaw_separation(smallerimg,sep)
    draw.draw_jaw_separation(smallerimg,lowery + sep)
    colimgtf = io.greyscale_to_colour(smallerimg)
    io.show_on_screen(colimgtf,1)
    return sep

# Gets the initial pose transformation to apply to the mean shape to start the ASM algorithm    
def get_initial_transformation(img,meanlm,orient):
    #meanlm = np.reshape(meanlm,(meanlm.size/2,2),'F')
    half = meanlm.size / 2
    sep = get_jaw_separation(img)
    scalefactor = 900
    scale = scalefactor * np.sqrt(meanlm.size/80)
    rot = 0
    #tx = 360
    tx = 390
    if orient == 0:
        ty = sep + scale * np.min(meanlm[half:])
    elif orient == 1:
        ty = sep - scale * np.min(meanlm[half:])
    else:
        print "Only up and down are supported as of yet."
    ty = ty - 10
    return tx, ty, scale, rot
