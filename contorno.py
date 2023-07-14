import cv2 

imagen = cv2.imread('C:/Users/SWIFT 3/source/Phyton/Entrenamiento/Monedas_Contorno/contorno.jpg')
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
tipo, umbral = cv2.threshold(grises,100,255,cv2.THRESH_BINARY)
contorno, jerarquia = cv2.findContours(umbral, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen,contorno,-1,(65, 105, 225),3)

#Mostrar
cv2.imshow('Imagen Original',imagen)
cv2.imshow('Imagen Grises', grises)
cv2.imshow('Imagen Umbral', umbral)
cv2.waitKey(0)
cv2.destroyAllWindows()