import epipolar
import draw
import camera
import numpy as np
import numpy.linalg as la
import gflags
import pylab as pl
import scipy
import scipy.io
import scipy.misc
import cv2
import cv2.cv as cv
import matplotlib.cm as cm
import signal
import random

signal.signal(signal.SIGINT, signal.SIG_DFL)

np.set_printoptions(suppress=True, precision=6)

def load_dinosaur():
    imgidx1 = 0
    imgidx2 = 1
    ## ----- Data loading (Vgg Dinosaur)
    img1 = cv2.imread("data/dino/viff.%03i.ppm"%imgidx1, cv.IPL_DEPTH_8U)
    img2 = cv2.imread("data/dino/viff.%03i.ppm"%imgidx2, cv.IPL_DEPTH_8U)

    Ps = scipy.io.loadmat("data/dino/dino_Ps.mat")['P']
    # TODO: Looks like loadmat require some byte order conversion...
    P1 = np.array(Ps[:,imgidx1][0], dtype=float).byteswap().newbyteorder()
    P2 = np.array(Ps[:,imgidx2][0], dtype=float).byteswap().newbyteorder()

    K1, R1 = camera.KR_from_P(P1)
    K2, R2 = camera.KR_from_P(P2)

    assert np.allclose(K1, K2)
    K = K1

    # Assume no distortion
    d = None

    matches = np.loadtxt("data/dino/viff.xy.txt")

    # Get matches between images 000 and 002
    choices = np.logical_and(matches[:,2*imgidx1] != -1, matches[:,2*imgidx2] != -1)
    x1 = matches[choices,2*imgidx1:2*imgidx1+2].T
    x2 = matches[choices,2*imgidx2:2*imgidx2+2].T

    return img1, img2, x1, x2, K, d

def load_human():
    img1 = cv2.imread("data/human/image1.png", cv.IPL_DEPTH_8U)
    img2 = cv2.imread("data/human/image2.png", cv.IPL_DEPTH_8U)
    K = np.load("data/human/K.npy")
    d = np.load("data/human/d.npy")
    x1 = np.load("data/human/x1.npy")
    x2 = np.load("data/human/x2.npy")
    return img1, img2, x1, x2, K, d

img1, img2, x1, x2, K, d = load_human()

#draw.draw_matches(img1, img2, x1, x2)

## ---- Fundamental matrix computation
F, status = cv2.findFundamentalMat(x1.T, x2.T)
status = np.ravel(status)

# Select only inliers
inliers = np.flatnonzero(status)
x1 = x1[:,inliers]
x2 = x2[:,inliers]

#draw.draw_matches(img1, img2, x1, x2)

## ---- Rectification based on found fundamental matrix
def rectify_images(img1, x1, img2, x2, K, d, F, shearing=False):
    imsize = (img1.shape[1], img1.shape[0])
    H1, H2 = epipolar.rectify_uncalibrated(x1, x2, F, imsize)

    #x1_1d = np.empty((2*x1.shape[1],), dtype=float)
    #x1_1d[0::2] = x1[0,:]
    #x1_1d[1::2] = x1[1,:]

    #x2_1d = np.empty((2*x2.shape[1],), dtype=float)
    #x2_1d[0::2] = x2[0,:]
    #x2_1d[1::2] = x2[1,:]

    #success, cvH1, cvH2 = cv2.stereoRectifyUncalibrated(x1_1d, x2_1d, F, imsize)


    if shearing:
        S = epipolar.rectify_shearing(H1, H2, imsize)
        H1 = S.dot(H1)

    rH = la.inv(K).dot(H1).dot(K)
    lH = la.inv(K).dot(H2).dot(K)

    # TODO: lRect or rRect for img1/img2 ??
    map1x, map1y = cv2.initUndistortRectifyMap(K, d, rH, K, imsize,
                                               cv.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K, d, lH, K, imsize,
                                               cv.CV_16SC2)

    # Convert the images to RGBA (add an axis with 4 values)
    img1 = np.tile(img1[:,:,np.newaxis], [1,1,4])
    img1[:,:,3] = 255
    img2 = np.tile(img2[:,:,np.newaxis], [1,1,4])
    img2[:,:,3] = 255

    rimg1 = cv2.remap(img1, map1x, map1y,
                      interpolation=cv.CV_INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0,0,0,0))
    rimg2 = cv2.remap(img2, map2x, map2y,
                      interpolation=cv.CV_INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0,0,0,0))

    # Put a red background on the invalid values
    # TODO: Return a mask for valid/invalid values
    # TODO: There is aliasing hapenning on the images border. We should
    # invalidate a margin around the border so we're sure we have only valid
    # pixels
    rimg1[rimg1[:,:,3] == 0,:] = (255,0,0,255)
    rimg2[rimg2[:,:,3] == 0,:] = (255,0,0,255)

    return rimg1, rimg2

pl.figure()
pl.suptitle('Without shearing transform')
rimg1, rimg2 = rectify_images(img1, x1, img2, x2, K, d, F, shearing=False)
epipolar.show_rectified_images(rimg1, rimg2)

pl.figure()
pl.suptitle('With shearing transform')
rimg1, rimg2 = rectify_images(img1, x1, img2, x2, K, d, F, shearing=True)
epipolar.show_rectified_images(rimg1, rimg2)

pl.show()

##
