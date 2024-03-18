import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('Mariposa.jpg', cv.IMREAD_GRAYSCALE)
 
# Crea un objeto FAST con los valores predeterminados
fast = cv.FastFeatureDetector_create()
 
# Encuentra y dibuja las regiones de interes
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
 
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
 
cv.imshow('fast_true.png', img2)
 
# Desactiva nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
 
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
 
img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

cv.imshow('fast_false.png', img3)
cv.waitKey(0)