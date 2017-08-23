# -*- coding: utf-8 -*-
import cv2
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
    
    return sep, upperyf, loweryf

def get_centralisation_old(img, yval):
    start = 300
    eind = 401
    
    reepje = np.copy(img[yval,start:eind])
    reepgrad = gl.get_gradients_raw(np.asarray(reepje))
    
    # Geef voorrang aan pieken dicht bij het centrum
    mid = (eind-start) / 2
    for i in range(mid):
        reepje[i] *= 0.75 + 0.25*(float(mid-i)/mid)
        reepje[mid + i] *= 0.75 + 0.25*(float(i)/mid)
        reepgrad[i] *= (float(i)/mid)
        reepgrad[mid + i] *= (float(mid-i)/mid)
    #draw.vec2graph(reepje)
    #draw.vec2graph(reepgrad)
    
    #x_offset = np.argmax(reepgrad) - mid
    x_offset = np.argmin(reepje) - mid
    
    print x_offset
    return x_offset

def get_centralisation(img):
    lines, xoffset = hough(img)
    #coli = io.greyscale_to_colour(img)
    #draw.draw_hough_lines(coli,lines)
    #io.show_on_screen(io.greyscale_to_colour(img),1)
    return xoffset

def hough(img):
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,15)
    img = io.greyscale_to_colour(edges)
    xes = np.array([])
    selected = np.array([])
    for rho,theta in lines[0]:
        if theta * 57 > 2:
            continue
        a = np.cos(theta)
        x0 = a*rho
        if (abs(xes - x0) < 20).any() and xes.size > 0:
             continue
        xes = np.append(xes,x0)
        selected = np.append(selected,[rho,theta])
    
    selected = np.reshape(selected,(selected.size /2, 2))
    closest = np.argmin(abs(xes - img[0,:,0].size /2 ))
    xoffset = xes[closest] - img[0,:,0].size /2
    return selected, xoffset
    
    
# Gets the initial pose transformation to apply to the mean shape to start the ASM algorithm    
def get_initial_transformation(img,meanlm,orient):
    #meanlm = np.reshape(meanlm,(meanlm.size/2,2),'F')
    half = meanlm.size / 2
    sep, upper, lower = get_jaw_separation(img)
    if orient == 0:
        scalefactor = 900
        scale = scalefactor * np.sqrt(meanlm.size/80)
        ty = upper + scale * np.min(meanlm[half:]) + 10
        tx = 350 + get_centralisation(img[250 + sep-200: 250 + sep,250:450])
        rot = 0
    elif orient == 1:
        scalefactor = 750
        scale = scalefactor * np.sqrt(meanlm.size/80)
        ty = lower - scale * np.min(meanlm[half:])
        tx = 350 + get_centralisation(img[250 + sep: 250 + sep + 200,250:450])
        rot = 0.08
    else:
        print "Only up and down are supported as of yet."
    return tx, ty, scale, rot
