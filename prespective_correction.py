import math
import cv2
import scipy.spatial.distance
import numpy as np
from rectification import find_corners, find_4corners, rectification_default

img = cv2.imread('data/prova6.png')
def perspective_correction(img):
    (rows, cols, _) = img.shape

    #image center
    u0 = (cols)/2.0
    v0 = (rows)/2.0

    #detected corners on the original image
    approx_corners = find_corners(img)
    tl, bl, br, tr = find_4corners(approx_corners)

    pp = []
    pp.append((67,74)) #    tl (0)
    pp.append((270,64)) #   tr (1)
    pp.append((10,344)) #   bl (2)
    pp.append((343,331)) #  br (3)

    #widths and heights of the projected image
    w1 = scipy.spatial.distance.euclidean(tl, tr)
    w2 = scipy.spatial.distance.euclidean(bl, br)

    h1 = scipy.spatial.distance.euclidean(tl, bl)
    h2 = scipy.spatial.distance.euclidean(tr, br)

    w = max(w1,w2)
    h = max(h1,h2)

    #visible aspect ratio
    ar_vis = float(w)/float(h)

    #make numpy arrays and append 1 for linear algebra
    m1 = np.array((tl[0], tl[1],1)).astype('float32')
    m2 = np.array((tr[0], tr[1],1)).astype('float32')
    m3 = np.array((bl[0], bl[1],1)).astype('float32')
    m4 = np.array((br[0], br[1],1)).astype('float32')

    #calculate the focal disrance
    k2 = np.dot(np.cross(m1,m4),m3) / np.dot(np.cross(m2,m4),m3)
    k3 = np.dot(np.cross(m1,m4),m2) / np.dot(np.cross(m3,m4),m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = math.sqrt(np.abs((1.0/(n23*n33)) * ((n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0*u0) + (n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0))))

    A = np.array([[f,0,u0], [0,f,v0], [0,0,1]]).astype('float32')

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    #calculate the real aspect ratio
    ar_real = math.sqrt(np.dot(np.dot(np.dot(n2,Ati),Ai),n2)/np.dot(np.dot(np.dot(n3,Ati),Ai),n3))

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    pts1 = np.array([tl, tr, bl, br]).astype('float32')
    pts2 = np.float32([[0,0],[W,0],[0,H],[W,H]])

    #project the image with the new w/h
    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(W,H))
    return dst