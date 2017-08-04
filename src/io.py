# -*- coding: utf-8 -*-
import cv2
import numpy as np

def show_on_screen(img,scale=1):
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width, _ = img.shape;
    cv2.resizeWindow('image', width / scale, height / scale)
    cv2.imshow('image',img);

def get_objectspace_img():
    img = np.zeros((512,512,3), np.uint8)
    img[:] = (255, 255, 255)
    return img

def get_img(i):
    if i < 10:
        img = cv2.imread("data/Radiographs/0"+str(i)+".tif")
    else:
        img = cv2.imread("data/Radiographs/"+str(i)+".tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[500:1430,1150:1850]
    
    return img
    
def get_all_enhanced_img(excepti=-1):
    edges = np.array([])
    for i in range(14):
        if (i+1 == excepti):
            continue
        else:
            imgi = get_img(i+1)
            imgi = pp.apply_filter_train(imgi)
            imgi = pp.apply_sobel(imgi)
            if(len(edges) == 0):
                edges = np.asarray(imgi)
            else:
                edges = np.append(edges,imgi)
    
    return edges.reshape(13,930,700)
    

# Returns a landmark file as a list of the form 
# (x0, x1, ..., y0, y1, ...)^T
def read_landmark(lm_file):
    data = np.array(map(lambda x: float(x.strip()),open(lm_file).readlines()))
    return np.reshape(data,(40,2)).flatten('F').T
    
# Returns a mirrored landmark file as a list of the form 
# (x0, x1, ..., y0, y1, ...)^T
def read_mirrored_landmark(lm_file):
    pts = np.reshape(map(lambda x: float(x.strip()),open(lm_file).readlines()),(40,2))
    for i in range(40):
        pts[i,0] = 3022 + pts[i,0]
    return pts.flatten('F').T


# Returns a collection of landmark files for a specific radiograph as a matrix 
# of the form 
# (x00, x10, ..., xn0)
# (x01, x11, ..., xn1)
# (..., ..., ..., ...)
# (y00, y10, ..., yn0)
# (y01, y11, ..., yn1)
# (..., ..., ..., ...)
def read_all_landmarks_by_img(imgi):
    lsx = np.array([])
    lsy = np.array([])
    for i in range(8):
        if imgi < 14:
            tooth = read_landmark('data/Landmarks/landmarks'+str(imgi)+'-'+str(i+1)+'.txt')
        else:
            tooth = read_mirrored_landmark('data/Landmarks/landmarks'+str(imgi)+'-'+str(i+1)+'.txt')
        half = tooth.size / 2
        lsx = np.concatenate((lsx,tooth[:half]))
        lsy = np.concatenate((lsy,tooth[half:]))
    result = np.concatenate((lsx,lsy))
    return np.reshape(result,(result.size,1))


# Returns a collection of landmark files for a specific radiograph as a matrix 
# of the form 
# (x00, x10, ..., xn0)
# (x01, x11, ..., xn1)
# (..., ..., ..., ...)
# (y00, y10, ..., yn0)
# (y01, y11, ..., yn1)
# (..., ..., ..., ...)
def read_all_landmarks_by_tooth(toothi):
    ls = []
    for i in range(14):
        ls.append(read_landmark('data/Landmarks/landmarks'+str(i+1)+'-'+str(toothi + 1)+'.txt'))
    for i in range(14):
        ls.append(read_mirrored_landmark('data/Landmarks/landmarks'+str(i+15)+'-'+str(toothi + 1)+'.txt'))
    return np.array(ls).T


# Returns a collection of landmark files for a specific radiograph as a matrix 
# of the form 
#
#      img0 img1      imgn
# t1  (x00, x10, ..., xn0)
#     (x01, x11, ..., xn1)
#     (..., ..., ..., ...)
# t2  (x00, x10, ..., xn0)
#     (x01, x11, ..., xn1)
#     (..., ..., ..., ...)
# t1  (y00, y10, ..., yn0)
#     (y01, y11, ..., yn1)
#     (..., ..., ..., ...)
#t2   (y00, y10, ..., yn0)
#     (y01, y11, ..., yn1)
#     (..., ..., ..., ...)
#
# number of rows is a multiple of 80 (80 coordinates per landmark file)
def read_all_landmarks_by_orientation(toothies):
    ls = []
    #for l in range(28):
     #   ls.append([])
    
    for i in range(28):
        imgteeth = read_all_landmarks_by_img(i + 1)
        half = imgteeth.size / 2
        orientsplit = half / 2
        orientteeth = np.array([])
        if toothies == 0:
            orientteeth = np.concatenate(((imgteeth[:orientsplit]),imgteeth[half:half+orientsplit]))
        elif toothies == 1:
            orientteeth = np.concatenate(((imgteeth[orientsplit:half]),imgteeth[orientsplit + half:]))
        else: 
            orientteeth = imgteeth
        ls.append(orientteeth.flatten())
    
    return np.array(ls).T
            
    