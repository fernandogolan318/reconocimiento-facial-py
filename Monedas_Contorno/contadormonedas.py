import cv2
import numpy as np
import tkinter as tk


valorGauss=1
valorKernel=7
original=cv2.imread('C:/Users/SWIFT 3/source/Phyton/Entrenamiento/Monedas_Contorno/monedassoles.jpg')
gris=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
gauss=cv2.GaussianBlur(gris, (valorGauss,valorGauss), 0)
canny=cv2.Canny(gauss, 60,100)
kernel=np.ones((valorKernel,valorKernel),np.uint8)
cierre=cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

contornos, jerarquía=cv2.findContours(cierre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("monedas encontradas: {}".format(len(contornos)))
cv2.drawContours(original, contornos, -1, (0,0,255),2)
# Mostrar resultados
cv2.imshow("Grises",gris)
cv2.imshow("gauss",gauss)
cv2.imshow("canny",canny)
cv2.imshow("cierre",cierre)
cv2.imshow("Resultado", original)
cv2.waitKey(0)

ventana = tk.Tk()
ventana.title("Resultado")
etiqueta = tk.Label(ventana, text="monedas encontradas: {}".format(len(contornos)))
etiqueta.pack(padx=20, pady=20)
ventana.mainloop()