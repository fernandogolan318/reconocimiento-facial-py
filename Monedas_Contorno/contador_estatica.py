import cv2
import numpy as np

valorGauss = 3
valorKernel = 3
original = cv2.imread('C:/Users/SWIFT 3/source/Phyton/Entrenamiento/Monedas_Contorno/monedas.jpg')
grises = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(grises,(valorGauss,valorKernel),0 )

cv2.imshow("Grises", grises)
cv2.imshow("Gauss", gauss)

cv2.waitKey(0)
