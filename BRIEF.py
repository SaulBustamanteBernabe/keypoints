import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('Mariposa.jpg', cv.IMREAD_GRAYSCALE)
 
# Inicializa el detecotr FAST
star = cv.xfeatures2d.StarDetector_create()
 
# Inicializa el extractor BRIEF
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
 
# Encuentra las regiones de interes con STAR
kp = star.detect(img,None)
 
# Calcula los descriptores con BRIEF
kp, des = brief.compute(img, kp)
 
print( brief.descriptorSize() )
print( des.shape )

# Dibuja las regiones de interes
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

cv.imshow('brief', img2)
cv.waitKey(0)