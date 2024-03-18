import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('Mariposa.jpg', cv.IMREAD_GRAYSCALE)
 
# inicializa el detector ORB
orb = cv.ORB_create()
 
# Busca las regiones de interes con ORB
kp = orb.detect(img,None)
 
# Calcula los descriptores con ORB
kp, des = orb.compute(img, kp)
 
# Dibuja unicamente el lugar de las regiones de interes, sin tamaño ni orientación
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()