# -*- coding: utf-8 -*-
import cv2
import numpy as np
import preprocess as pp

# Show an image on screen
def show_on_screen(img,scale=1):
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width, _ = img.shape;
    cv2.resizeWindow('image', width / scale, height / scale)
    cv2.imshow('image',img);
    cv2.waitKey(0);

# Show a greyscale image on screen    
def show_on_screen_greyscale(img,scale=1):
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    height, width = img.shape;
    cv2.resizeWindow('image', width / scale, height / scale)
    cv2.imshow('image',img);

# Transform the color space from greyscale to BGR    
def greyscale_to_colour(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Return a white image that allows the representation of shapes in object space
def get_objectspace_img():
    img = np.zeros((512,512,3), np.uint8)
    img[:] = (255, 255, 255)
    return img

# Get a radiograph image and crop it
def get_img(i):
    if i < 10:
        img = cv2.imread("data/Radiographs/0"+str(i)+".tif")
    else:
        img = cv2.imread("data/Radiographs/"+str(i)+".tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[500:1430,1150:1850]
    
    return img

# Get an enhanced image and apply the sobel operator on it    
def get_gradient_img(i):
    imgi = get_img(i)
    imgi = pp.apply_filter_train(imgi)
    imgi = pp.apply_sobel(imgi)
    return imgi
 
# Get a radiograph image, crop it and apply the filter train on it to enhance it   
def get_enhanced_img(i):
    imgi = get_img(i)
    imgi = pp.apply_filter_train(imgi)
    return imgi
 
# Get all gradient images except the one with the given index to generate
#  the grey level model   
def get_all_gradient_img(excepti=-1):
    edges = np.array([])
    for i in range(28):
        if (i+1 == excepti or i+1 == excepti + 14):
            continue
        else:
            imgi = None
            if i >= 14:
                imgi = cv2.flip(get_gradient_img(i - 13),1)
            else: 
                imgi = get_gradient_img(i+1)
            if(len(edges) == 0):
                edges = np.asarray(imgi)
            else:
                edges = np.append(edges,imgi)
    
    return edges.reshape(26,930,700)
    

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
  
def get_result_eval():
    return np.array([[  7.59230910e+01,   7.63560709e+01,   7.54901110e+01,   8.69257515e-01,
    1.30742485e-01,   6.46392225e-01,   3.53607775e-01],
 [  7.66737345e+01,   8.17123871e+01,   7.16350819e+01,   8.62217473e-01,
    1.37782527e-01,   6.50341890e-01,   3.49658110e-01],
 [  8.10806624e+01,   9.78480611e+01,   6.43132637e+01,   8.59201118e-01,
    1.40798882e-01,   6.44395542e-01,   3.55604458e-01],
 [  8.32255824e+01,   6.68314466e+01,   9.96197181e+01,   5.82779188e-01,
    4.17220812e-01,   6.28661456e-01,   3.71338544e-01],
 [  8.64498185e+01,   7.84022935e+01,   9.44973435e+01,   5.44166123e-01,
    4.55833877e-01,   6.22903202e-01,   3.77096798e-01],
 [  1.18957186e+02,   1.53614781e+02,   8.42995912e+01,   4.05013096e-01,
    5.94986904e-01,   4.73193518e-01,   5.26806482e-01],
 [  3.61418571e+01,   3.71253560e+01,   3.51583583e+01,   8.12302201e-01,
    1.87697799e-01,   8.60000612e-01,   1.39999388e-01],
 [  3.76975724e+01,   4.03234149e+01,   3.50717299e+01,   8.10214020e-01,
    1.89785980e-01,   8.55101309e-01,   1.44898691e-01],
 [  6.41260929e+01,   9.31804560e+01,   3.50717299e+01,   7.28361164e-01,
    2.71638836e-01,   8.07651630e-01,   1.92348370e-01],
 [  3.15519040e+01,   3.60677192e+01,   2.70360887e+01,   8.07613732e-01,
    1.92386268e-01,   7.71413387e-01,   2.28586613e-01],
 [  3.69559551e+01,   4.68758215e+01,   2.70360887e+01,   7.58402156e-01,
    2.41597844e-01,   7.35666400e-01,   2.64333600e-01],
 [  6.84886162e+01,   1.09941144e+02,   2.70360887e+01,   7.16686372e-01,
    2.83313628e-01,   6.62091509e-01,   3.37908491e-01],
 [  1.33931442e+02,   2.28025976e+02,   3.98369075e+01,   2.86109936e-01,
    7.13890064e-01,   2.70690488e-01,   7.29309512e-01],
 [  1.37908039e+02,   2.35138088e+02,   4.06779896e+01,   2.79181334e-01,
    7.20818666e-01,   2.76944152e-01,   7.23055848e-01],
 [  1.56405409e+02,   2.71868867e+02,   4.09419504e+01,   2.82129059e-01,
    7.17870941e-01,   2.71686742e-01,   7.28313258e-01],
 [  7.88516412e+01,   2.22015871e+01,   1.35501695e+02,   5.56271923e-01,
    4.43728077e-01,   5.61701809e-01,   4.38298191e-01],
 [  7.98947816e+01,   2.42133439e+01,   1.35576219e+02,   5.49837375e-01,
    4.50162625e-01,   5.57536237e-01,   4.42463763e-01],
 [  9.21863301e+01,   4.86798648e+01,   1.35692795e+02,   5.08275419e-01,
    4.91724581e-01,   5.01812890e-01,   4.98187110e-01],
 [  4.00655082e+01,   4.63983806e+01,   3.37326359e+01,   8.28414615e-01,
    1.71585385e-01,   8.02077356e-01,   1.97922644e-01],
 [  3.72153999e+01,   4.27786963e+01,   3.16521035e+01,   8.27874133e-01,
    1.72125867e-01,   8.15033448e-01,   1.84966552e-01],
 [  3.36789381e+01,   3.84177873e+01,   2.89400889e+01,   8.23728988e-01,
    1.76271012e-01,   8.01485951e-01,   1.98514049e-01],
 [  8.91239780e+01,   1.55548469e+02,   2.26994869e+01,   8.33787093e-01,
    1.66212907e-01,   5.74333987e-01,   4.25666013e-01],
 [  9.64097688e+01,   1.70016802e+02,   2.28027351e+01,   8.43034387e-01,
    1.56965613e-01,   5.40864752e-01,   4.59135248e-01],
 [  1.20458268e+02,   2.09686801e+02,   3.12297348e+01,   8.06569933e-01,
    1.93430067e-01,   4.72355515e-01,   5.27644485e-01],
 [  5.56517405e+01,   4.10060416e+01,   7.02974394e+01,   6.92963652e-01,
    3.07036348e-01,   6.46733357e-01,   3.53266643e-01],
 [  7.45733359e+01,   4.80191679e+01,   1.01127504e+02,   6.96460924e-01,
    3.03539076e-01,   6.00248526e-01,   3.99751474e-01],
 [  1.34707799e+02,   7.30976732e+01,   1.96317925e+02,   5.91275149e-01,
    4.08724851e-01,   5.59701895e-01,   4.40298105e-01],
 [  2.70938242e+01,   3.33464490e+01,   2.08411993e+01,   8.74196220e-01,
    1.25803780e-01,   8.73600428e-01,   1.26399572e-01],
 [  3.07606328e+01,   3.81789716e+01,   2.33422940e+01,   8.41639062e-01,
    1.58360938e-01,   8.13917700e-01,   1.86082300e-01],
 [  4.64224081e+01,   6.95025222e+01,   2.33422940e+01,   8.32659457e-01,
    1.67340543e-01,   7.28139555e-01,   2.71860445e-01],
 [  4.77972173e+01,   6.90225937e+01,   2.65718409e+01,   8.40350756e-01,
    1.59649244e-01,   7.89780908e-01,   2.10219092e-01],
 [  5.32962148e+01,   7.98466711e+01,   2.67457586e+01,   8.30957993e-01,
    1.69042007e-01,   7.82130414e-01,   2.17869586e-01],
 [  7.51010390e+01,   1.18233214e+02,   3.19688643e+01,   7.77482011e-01,
    2.22517989e-01,   7.26270802e-01,   2.73729198e-01],
 [  1.01369532e+02,   1.09714678e+02,   9.30243861e+01,   7.70931420e-01,
    2.29068580e-01,   4.55121279e-01,   5.44878721e-01],
 [  1.18830209e+02,   1.44010925e+02,   9.36494935e+01,   6.80786165e-01,
    3.19213835e-01,   3.75569749e-01,   6.24430251e-01],
 [  1.46471647e+02,   2.04934625e+02,   8.80086683e+01,   6.64827659e-01,
    3.35172341e-01,   3.58550349e-01,   6.41449651e-01],
 [  3.57069110e+01,   3.91226382e+01,   3.22911837e+01,   8.19290982e-01,
    1.80709018e-01,   8.56318584e-01,   1.43681416e-01],
 [  3.21158148e+01,   3.08548771e+01,   3.33767525e+01,   8.41464447e-01,
    1.58535553e-01,   8.55900236e-01,   1.44099764e-01],
 [  2.62791829e+01,   1.84392615e+01,   3.41191042e+01,   8.73081189e-01,
    1.26918811e-01,   8.53433864e-01,   1.46566136e-01],
 [  5.13265987e+01,   5.12997388e+01,   5.13534585e+01,   7.24964757e-01,
    2.75035243e-01,   8.37098147e-01,   1.62901853e-01],
 [  6.33283170e+01,   5.75621343e+01,   6.90944997e+01,   7.16423171e-01,
    2.83576829e-01,   7.89722762e-01,   2.10277238e-01],
 [  9.82676668e+01,   7.89988895e+01,   1.17536444e+02,   6.33748310e-01,
    3.66251690e-01,   7.35715030e-01,   2.64284970e-01]])

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
        if imgi <= 14:
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
def read_all_landmarks_by_orientation(toothies, excepti=-100):
    ls = []
    #for l in range(28):
     #   ls.append([])
    
    for i in range(28):
        if (i+1 == excepti or i+1 == excepti + 14):
            continue
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
            
    