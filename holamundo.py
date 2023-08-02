from PIL import Image

# Cargar la imagen
imagen = Image.open("C:/Users/SWIFT 3/source/Phyton/Entrenamiento/Monedas_Contorno/img/monedas.jpg")

# Obtener el tamaño de la imagen
ancho, altura = imagen.size

# Imprimir el tamaño de la imagen
print('El tamaño de la imagen es:', ancho, 'x', altura)