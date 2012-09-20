import numpy as np
import pylab as pl
import cv2

def side_by_side_images(img1, img2):
    height = max(img1.shape[0], img2.shape[0])
    xoff = img1.shape[1]
    vimg = np.zeros((height, img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    vimg[0:img1.shape[0], 0:img1.shape[1]] = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    vimg[0:img2.shape[0], xoff:xoff+img2.shape[1]] = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return vimg

def draw_matches(img1, img2, x1, x2):
    vimg = side_by_side_images(img1, img2)
    xoff = img1.shape[1]
    pl.figure()
    pl.title('matches')
    pl.imshow(vimg)
    pl.gca().autoscale(False)

    for i in xrange(x1.shape[1]):
        p1 = x1[:,i]
        p2 = x2[:,i] + [xoff,0]
        pl.plot([p1[0], p2[0]], [p1[1], p2[1]], '+-')
