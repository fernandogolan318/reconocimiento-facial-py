import cv2
import numpy as np
import tkinter as tk


def cargar_imagen(ruta):
    """
    Función para cargar una imagen desde una ruta.
    """
    imagen = cv2.imread(ruta)
    return imagen


def convertir_a_grises(imagen):
    """
    Función para convertir una imagen a escala de grises.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return gris


def aplicar_filtro_gaussiano(imagen, valor_gauss):
    """
    Función para aplicar un filtro gaussiano a una imagen.
    """
    gauss = cv2.GaussianBlur(imagen, (valor_gauss, valor_gauss), 0)
    return gauss


def detectar_bordes(imagen):
    """
    Función para detectar bordes en una imagen utilizando el algoritmo de Canny.
    """
    canny = cv2.Canny(imagen, 60, 100)
    return canny


def aplicar_operacion_morfologica(imagen, valor_kernel):
    """
    Función para aplicar una operación morfológica a una imagen.
    """
    kernel = np.ones((valor_kernel, valor_kernel), np.uint8)
    cierre = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
    return cierre


def encontrar_contornos(imagen):
    """
    Función para encontrar los contornos en una imagen.
    """
    contornos, jerarquia = cv2.findContours(imagen.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos


def dibujar_contornos(imagen, contornos):
    """
    Función para dibujar los contornos en una imagen.
    """
    cv2.drawContours(imagen, contornos, -1, (0, 0, 255), 2)
    return imagen


def mostrar_imagenes(imagenes):
    """
    Función para mostrar varias imágenes.
    """
    for nombre, imagen in imagenes.items():
        cv2.imshow(nombre, imagen)
    cv2.waitKey(0)


def mostrar_resultado(contornos):
    """
    Función para mostrar el resultado en una ventana de tkinter.
    """
    ventana = tk.Tk()
    ventana.title("Resultado")
    etiqueta = tk.Label(ventana, text="Monedas encontradas: {}".format(len(contornos)))
    etiqueta.pack(padx=20, pady=20)
    ventana.mainloop()


# Cargar imagen original
ruta = 'C:/Users/SWIFT 3/source/Phyton/Entrenamiento/Monedas_Contorno/img/monedas.jpg'
original = cargar_imagen(ruta)

# Convertir a escala de grises
gris = convertir_a_grises(original)

# Aplicar filtro Gaussiano
valor_gauss = 1
gauss = aplicar_filtro_gaussiano(gris, valor_gauss)

# Detectar bordes
canny = detectar_bordes(gauss)

# Aplicar operación morfológica
valor_kernel = 7
cierre = aplicar_operacion_morfologica(canny, valor_kernel)

# Encontrar contornos
contornos = encontrar_contornos(cierre)

# Dibujar contornos en imagen original
resultado = dibujar_contornos(original.copy(), contornos)

# Mostrar imágenes
imagenes = {
    "Grises": gris,
    "Gauss": gauss,
    "Canny": canny,
    "Cierre": cierre,
    "Resultado": resultado
}
mostrar_imagenes(imagenes)

# Mostrar resultado en ventana de tkinter
mostrar_resultado(contornos)