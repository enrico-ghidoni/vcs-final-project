import imutils
import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import math
import scipy.spatial.distance
#import the necessary library :)


def show_image(image):
    """
    Function to plot an image
    Keyword arguments: image
    Return: None
    """
    cv2.imshow("Immagine",image)
    cv2.waitKey(0)
    return None


def find_4corners(approx_corners):
    """
    Function able to find the 4 external corners in a set of points.
    Keyword arguments: list of points
    Return: list of 4 more distant corners by each other
    """
    if approx_corners is None:
        print("\n ERROR 2: corners not found \n")
        return None
    #  find the corner points of the contours and print it
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    # find the most extreme corners, only 4 corner points
    x = [a[0] for a in approx_corners]
    y = [a[1] for a in approx_corners]
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest and compute the difference between 
    # the points. the top-right will have the minumum difference and 
    # the bottom-left will have the maximum difference
    diff = np.array(x) - np.array(y) 
    addition = np.array(x) + np.array(y)
    tl = approx_corners[np.argmin(addition)] # top left point
    br = approx_corners[np.argmax(addition)] # bottom right point
    bl = approx_corners[np.argmin(diff)] # bottom left point
    tr = approx_corners[np.argmax(diff)] # top right point
    
    #print('\nAll the corner points are: ' + str([tl,bl,br,tr]))
    return [tl,bl,br,tr]


def find_corners(image):
    
    """
    Function to find all the corners in an image, not used in this case
    Keyword arguments: image
    Return: list of apporximated corners
    """
    if image is None:
        print("\n ERROR 1: image not found or missing \n")
        return None
    # grayscale image and blurring
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # threshold the image, then perform a series of 
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)[1] #aleatory param
    thresh = cv2.dilate(thresh, None, iterations=8) 
    # find contours in thresholded image, then grab the largest one
    cnt = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = imutils.grab_contours(cnt)
    cnt = max(cnt, key=cv2.contourArea)
    epsilon = 0.015*cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    return approx_corners


def find_edges(image):
    
    """
    Function to find the edges of an image
    Keyword arguments: image
    Return: list of approximated corners
    """
        
    if image is None:
        print("\n ERROR 1: image not found or missing \n")
        return None
    # grayscale image and blurring
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur = cv2.bilateralFilter(gray,10,17,17) # try with bilateral filter 
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.Canny(image,0,50) # try with canny edge detection, not always the best solution
    thresh = cv2.dilate(thresh, None, iterations=2) 
    # find contours in thresholded image, then grab the largest one
    cnt = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = imutils.grab_contours(cnt)
    cnt = max(cnt, key=cv2.contourArea)
    epsilon = 0.015*cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    
    return approx_corners


def rectification_db(image, image_db):

    """
    Function to implement the rectification in an image prevously matched with an images db
    Keyword arguments: image, image_db
    Return: original image, rectified image
    """
    approx_corners = find_edges(image)
    approx_corners_db = find_edges(image_db)
    #take the source points and destination points
    point_src = np.float32(find_4corners(approx_corners)) 
    point_dst = np.float32(find_4corners(approx_corners_db))
    # homography process and warp prespective images
    H, mask = cv2.findHomography(point_src, point_dst, cv2.RANSAC, 5.0) # mask unusued
    final_width = image_db.shape[0] 
    final_high = image_db.shape[1]
    rectified_image = cv2.warpPerspective(image, np.float32(H), (image_db.shape[1],image_db.shape[0]))
    rectified_image = exposure.rescale_intensity(rectified_image, out_range = (0, 255))
    
    return image, rectified_image


def perspective_correction(img):
    """
    Function to implement the rectification of an image. The formula was taken by a paper 
    on the web. Link: https://www.microsoft.com/en-us/research/publication/whiteboard-scanning-image-enhancement/?from=http%3A%2F%2
    Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fzhang%2Fpapers%2Ftr03-39.pdf
    Keyword arguments: image
    Return: rectified image
    """
    (rows, cols, _) = img.shape
    u0 = (cols)/2.0
    v0 = (rows)/2.0

    #detected corners on the original image
    approx_corners = find_edges(img)
    tl, bl, br, tr = find_4corners(approx_corners)
    print("CORNERS:" + str([tl,bl,br,tr]))
    """
    # TESTING: drawing all the corner points in the image
    cv2.putText(img, "LT", tuple(tl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, "LD", tuple(bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "RT", tuple(tr), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "RD", tuple(br), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    show_image(img) 
    """
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
    
    #calculate the focal distance
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
    try:
        f = math.sqrt(np.abs((1.0/(n23*n33)) * 
        ((n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0*u0) 
        + (n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0))))

        A = np.array([[f,0,u0], [0,f,v0], [0,0,1]]).astype('float32')

        At = np.transpose(A)
        Ati = np.linalg.inv(At)
        Ai = np.linalg.inv(A)
        #calculate the real aspect ratio
        ar_real = math.sqrt(np.dot(np.dot(np.dot(n2,Ati),Ai),n2)
            /np.dot(np.dot(np.dot(n3,Ati),Ai),n3))
        if ar_real < ar_vis:
            W = int(w)
            H = int(float(W / ar_real))       
        else:
            H = int(h)
            W = int(ar_real * H)
    except Exception:
        ar_real = ar_vis
        H = int(h)
        W = int(ar_real * H)
        
    pts1 = np.array([tl, tr, bl, br]).astype('float32')
    pts2 = np.float32([[0,0],[W,0],[0,H],[W,H]])
    #project the image with the new w/h
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img, M, (W,H))
    rectified_image = dst
    rectified_image = exposure.rescale_intensity(dst, out_range = (0, 255))
    return rectified_image
