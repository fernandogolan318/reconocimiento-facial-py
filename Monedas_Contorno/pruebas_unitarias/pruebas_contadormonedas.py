import sys
sys.path.append('C:/Users/SWIFT 3/source/Phyton/Entrenamiento/Monedas_Contorno')
import unittest
import cv2
import numpy as np
import tkinter as tk
from contadormonedas import cargar_imagen, convertir_a_grises, aplicar_filtro_gaussiano, detectar_bordes, aplicar_operacion_morfologica, encontrar_contornos, dibujar_contornos

class TestMoneda(unittest.TestCase):
    def setUp(self):
        self.ruta = "C:/Users/SWIFT 3/source/Phyton/Entrenamiento/Monedas_Contorno/img/monedas.jpg"

    def test_cargar_imagen(self):
        imagen = cargar_imagen(self.ruta)
        self.assertIsNotNone(imagen)
        self.assertIsInstance(imagen, np.ndarray)
        self.assertEqual(imagen.shape, (384, 700, 3))

    def test_convertir_a_grises(self):
        imagen = cargar_imagen(self.ruta)
        gris = convertir_a_grises(imagen)
        self.assertIsNotNone(gris)
        self.assertIsInstance(gris, np.ndarray)
        self.assertEqual(gris.shape, (384, 700))

    def test_aplicar_filtro_gaussiano(self):
        imagen = cargar_imagen(self.ruta)
        gris = convertir_a_grises(imagen)
        valor_gauss = 1
        gauss = aplicar_filtro_gaussiano(gris, valor_gauss)
        self.assertIsNotNone(gauss)
        self.assertIsInstance(gauss, np.ndarray)
        self.assertEqual(gauss.shape, (384, 700))

    def test_detectar_bordes(self):
        imagen = cargar_imagen(self.ruta)
        gris = convertir_a_grises(imagen)
        valor_gauss = 1
        gauss = aplicar_filtro_gaussiano(gris, valor_gauss)
        canny = detectar_bordes(gauss)
        self.assertIsNotNone(canny)
        self.assertIsInstance(canny, np.ndarray)
        self.assertEqual(canny.shape, (384, 700))

    def test_aplicar_operacion_morfologica(self):
        imagen = cargar_imagen(self.ruta)
        gris = convertir_a_grises(imagen)
        valor_gauss = 1
        gauss = aplicar_filtro_gaussiano(gris, valor_gauss)
        canny = detectar_bordes(gauss)
        valor_kernel = 7
        cierre = aplicar_operacion_morfologica(canny, valor_kernel)
        self.assertIsNotNone(cierre)
        self.assertIsInstance(cierre, np.ndarray)
        self.assertEqual(cierre.shape, (384, 700))

    def test_encontrar_contornos(self):
        imagen = cargar_imagen(self.ruta)
        gris = convertir_a_grises(imagen)
        valor_gauss = 1
        gauss = aplicar_filtro_gaussiano(gris, valor_gauss)
        canny = detectar_bordes(gauss)
        valor_kernel = 7
        cierre = aplicar_operacion_morfologica(canny, valor_kernel)
        contornos = encontrar_contornos(cierre)
        print(contornos)
        self.assertIsNotNone(contornos)
        self.assertIsInstance(contornos, list)
        self.assertGreater(len(contornos), 0)

    def test_dibujar_contornos(self):
        imagen = cargar_imagen(self.ruta)
        gris = convertir_a_grises(imagen)
        valor_gauss = 1
        gauss = aplicar_filtro_gaussiano(gris, valor_gauss)
        canny = detectar_bordes(gauss)
        valor_kernel = 7
        cierre = aplicar_operacion_morfologica(canny, valor_kernel)
        contornos = encontrar_contornos(cierre)
        resultado = dibujar_contornos(imagen.copy(), contornos)
        self.assertIsNotNone(resultado)
        self.assertIsInstance(resultado, np.ndarray)
        self.assertEqual(resultado.shape, (384, 700, 3))

    def test_mostrar_resultado(self):
        imagen = cargar_imagen(self.ruta)
        gris = convertir_a_grises(imagen)
        valor_gauss = 1
        gauss = aplicar_filtro_gaussiano(gris, valor_gauss)
        canny = detectar_bordes(gauss)
        valor_kernel = 7
        cierre = aplicar_operacion_morfologica(canny, valor_kernel)
        contornos = encontrar_contornos(cierre)
        ventana = tk.Tk()
        etiqueta = tk.Label(ventana, text="Monedas encontradas: {}".format(len(contornos)))
        etiqueta.pack(padx=20, pady=20)
        self.assertIsNotNone(ventana)
        self.assertIsInstance(ventana, tk.Tk)
        self.assertIsNotNone(etiqueta)
        self.assertIsInstance(etiqueta, tk.Label)


if __name__ == '__main__':
    # Crear una suite de pruebas
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMoneda)

    # Crear un corredor de pruebas y generar el reporte en un archivo de texto
    with open('reporte.txt', 'w') as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        result = runner.run(suite)