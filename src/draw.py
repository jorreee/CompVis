# -*- coding: utf-8 -*-
import cv2 as cv2
import numpy as np
import landmarks as lms
import matplotlib.pyplot as plt
import io as io

lengthn = 160
# Takes a column vector and interprets it as all x coordinates followed by all y coordinates, then draws a line between the points on the given image
def draw_contour(img,lm,color=(255,0,0),thicc=1):
#    for ind in range(lengthn-1):
    half = lm.size/2
    lpts = np.reshape(lm,(half,2),'F')
    for ind in range(len(lpts)):
        if (ind+1) % 40 == 0:
            cv2.line(img,
                (int(float(lpts[ind,0])),int(float(lpts[ind,1]))),
                (int(float(lpts[ind-39,0])),int(float(lpts[ind-39,1]))),
                color,
                thicc);
        else:
            cv2.line(img,
                (int(float(lpts[ind,0])),int(float(lpts[ind,1]))),
                (int(float(lpts[ind+1,0])),int(float(lpts[ind+1,1]))),
                color,
                thicc);
    
    return None
    
# Takes a matrix with landmarks as columns and draws them all on the given image
def draw_all_contours(img,ls):
    for lm in ls.T:
        #print lm.shape
        draw_contour(img,lm.T)
    return None

# Transform an object present in object space to make it visible in the object space image	
def make_object_space_visible(ls):
    return lms.transform_shape(ls,250,250,1003,0)

# Takes a matrix with aligned landmarks as column. Applies a scaling and translation transformation to them to be visible in an image and then draws the transformed landmarks on the given image
def draw_aligned_contours(img,ls):
    lst = np.copy(ls).T # alle lms zijn nu de rijen
    for i in range(lst.shape[0]): # ga over alle rijen
	lst[i] = make_object_space_visible(np.reshape(lst[i],(lst[i].size,1))).flatten()
    draw_all_contours(img,lst.T) # zet lms terug als columns en teken
    return None

# Draws the normals for landmarks , which all have to be situated in image space  
def draw_normals(img,lpts,norms):
    #Arbitrary scaling for visualizing the normals
    norms = -15 * norms  
    half = lpts.size/2
    lpts = np.reshape(lpts,(half,2),'F')
    norms = np.reshape(norms,(half,2),'F')
    for ind in range(160):
        cv2.line(img,
                (int(float(lpts[ind,0])),int(float(lpts[ind,1]))),
                (   int(float(norms[ind,0] + lpts[ind,0])),
                    int(float(norms[ind,1] + lpts[ind,1]))),
                (255,0,0),
                2);
    
    return None
    
def draw_slice(img,slicel):
    print slicel[0,1]
    for i in range(slicel.size / 2):
        img[slicel[1,i],slicel[0,i]] = 255
    
    return None

# Draws the jaw separation on an image   
def draw_jaw_separation(img,yval):
    for ind in range(len(img[1,:]) - 1):
        cv2.line(img,
                (ind,int(yval)),
                ((ind + 1),
                (int(yval))),
                (255,0,0),
                2);
    
    return None
    
def vec2graph(vec):
    plt.plot(vec)
    plt.show()
    cv2.waitKey(0)
    
def draw_hough_lines(img,lines):
    img = io.greyscale_to_colour(img)
    for rho,theta in lines:
        a = np.cos(theta)
        x0 = a*rho
        b = np.sin(theta)
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
    io.show_on_screen(img)
    
    return None

    
if __name__ == '__main__':
    vec2graph(np.array([1,2,3,4]))