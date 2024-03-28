# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:55:10 2024

@author: Jorge
"""
import cv2
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
from scipy.ndimage import gaussian_filter
# from scipy.signal import find_peaks

# from scipy.integrate import cumulative_trapezoid
# from scipy.linalg import lstsq
#
from numpy.fft import fft2, fftshift, ifft2, ifftshift
# from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
class Cargarimagenes:
    def __init__(self, img_rutas):
        self.img_rutas = img_rutas
        self.img_dict = {}
        self.textura = None
        self.upload_img()

    def upload_img(self):  # self es nuestro objeto instancia, lo llamamos
        # en la funcion ya que img_ruta es un atributo de # nuestro objeto!!
        '''
        input: diccionario con nuestras rutas

        output: diccionario con las imagenes y la textura irá aparte
        ya que no tiene por que tener las mismas cualidades que las demas imagenes
        :return:
        '''
        for [key, ruta] in self.img_rutas.items():  # iteramos en ruta --> despues de este bucle en el nuevo diccionario
            # habrá imagenes grayscale[0,255]
            # si la textura viene en RGB la cargamos asi
            if key == 'textura':
                textura = cv2.imread(ruta, cv2.IMREAD_COLOR)
                if textura is None:  # si imread no lee nada, devuelve None
                    print(f'La textura "{ruta} no se ha podido cargar. Mira que este la ruta correcta')
                    continue
                self.textura = cv2.cvtColor(textura, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
                # print(image.dtype)
                if image is None:
                    raise ValueError(f'La imagen "{key} no se pudo cargar. Mira que este la ruta correcta')
                self.img_dict[key] = image

class Procesarimagenes:
    def __init__(self, atributos_cargar):
        self.img_dict = atributos_cargar.img_dict
        self.textura = atributos_cargar.textura

        'atributos nuevos'
        self.ruido=None

        'aplicamos funciones'
        self.nivel_ruido()
        self.filtro(ver=False)
        self.aplicar_fourier(ver=False)


    def nivel_ruido(self) -> object:
        '''
        input: Nuestro objeto, mas en concreto el diccionario
        output: el ruido de nuestra imagen
        '''
        for key, image in self.img_dict.items():
            # if key != 'textura':  --> no hace falta, en esta nueva version no hay textura en img_dict
            self.ruido = np.std(image)
            print(f'El nivel de ruido de {key} es {self.ruido}')
            # return self.ruido #es necesario??'

    def filtro(self, sigma = 1, ver = True):
        '''
        Funcion que aplica un filtro gaussiano a nuestras imagenes
        Sabemos que tiene ruido gaussiano debido a los histogramas
        sigma: nivel de agresividad edl filtro
        '''
        for key, image in self.img_dict.items():
            img_no_filtrada = self.img_dict[key]
            self.img_dict[key] = gaussian_filter(image, sigma=sigma)
            print(f'La imagen {key} se ha filtrado correctamente')

            if ver==True:
                canva=plt.figure(figsize=(8,3))
                original=canva.add_subplot(121)
                original.imshow(img_no_filtrada, cmap ='gray')
                original.set_title(f'Original {key}')
                original.axis('off')

                filtrada=canva.add_subplot(122)
                filtrada.imshow(self.img_dict[key], cmap='gray')
                filtrada.set_title(f'Filtrada {key}')
                filtrada.axis('off')

                canva.tight_layout()
                plt.show(block=True)

    # def estimacion_radio(self):
    #     '''
    #     vamos a estimar el valor del el radio para la transofrmada de fourier en
    #     funcion del contenido de la imagen, ya que en imagenes SEM nos interesa
    #     preservar los detalles y eliminar ruido
    #     :return:
    #     '''
    #     r=70
    #     return r

    def transformada_fourier(self,image):
        t_fourier=fft2(image) # calcualo de la transformada rapida de fourier
        t_fourier=fftshift(t_fourier) #ponemos las frecuencias bajas en el medio del espectro
        return t_fourier

    def filtro_trans_inversa(self,t_fourier,r):
        row, col = t_fourier.shape
        mid_row,mid_col= row//2 , col//2 # mid_fila,mid_col= int(fila), int(col)

        #creamos el filtro passo-basso circular
        mask=np.zeros((row,col),np.uint8)
        centro=[mid_row,mid_col]
        x, y = np.ogrid[:row,:col] #mallado
        mask_area=(x-centro[0])**2 + (y-centro[1])**2 <=r**2 #(x-x0)^2+(y-y0)^2=r^2 de manual
        mask[mask_area]=1 #aplicamos el filtro y dejamos que pasen las frecuencias de dentro

        #aplicamos la mascara y la trans inversa
        t_fourier_mask=t_fourier*mask #aplicamos la mascara a nuestro espectro de frecuencias, filtramos las de fuera r
        inv_t_fourier=ifftshift(t_fourier_mask) # se invierte el espectro
        img_trans=np.abs(ifft2(inv_t_fourier))  #nos aseguramos que sea una imagen REAL
        return img_trans
    def aplicar_fourier(self,ver=True):
        '''
        Funcion que aplica la transformada de fourier sobre self.img_dict
        :param ver: ==True --> vemos la imagen transformada // ==False --> no hay plot
        '''
        for key, image in self.img_dict.items():
            t_fourier=self.transformada_fourier(image)
            r=self.estimacion_radio(t_fourier)
            # r=self.estimacion_radio()
            image_trans = self.filtro_trans_inversa(t_fourier,r)

            #metricas de calidad de la imagen
            ssim_val=ssim(image, image_trans, data_range = image.max() - image.min())
            psnr_val=psnr(image, image_trans, data_range = image.max() - image.min())
            print(f"{key} - SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.4f}")

            if ver == True:

                canva = plt.figure(figsize=(8, 3))
                canva.suptitle(f"{key} - SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.4f}",fontsize=14)

                original = canva.add_subplot(121)
                original.imshow(image, cmap='gray')
                original.set_title(f'Original {key}')
                original.axis('off')

                filtrada = canva.add_subplot(122)
                filtrada.imshow(image_trans, cmap='gray')
                filtrada.set_title(f'Transformada {key}')
                filtrada.axis('off')

                canva.tight_layout()
                plt.show(block=True)

            self.img_dict[key] = image_trans

    def calcular_varianzas(self):
        """
        Calcula la varianza de cada imagen en el diccionario img_dict y devuelve un diccionario
        de varianzas correspondiente a cada clave de imagen.
        """
        varianzas = {}
        for key, image in self.img_dict.items():
            varianzas[key] = np.var(image)
        return varianzas

    def ajustar_varianza_min_max(self):
        # Calcular varianzas de todas las imágenes
        varianzas = self.calcular_varianzas()
        valores_varianza = list(varianzas.values())

        # Ajustar los valores globales basados en el conjunto actual de imágenes
        self.varianza_min = min(valores_varianza)
        self.varianza_max = max(valores_varianza)

    def estimacion_radio(self, image):
        # Asegurar que 'varianza_min' y 'varianza_max' están definidos
        if not hasattr(self, 'varianza_min') or not hasattr(self, 'varianza_max'):
            self.ajustar_varianza_min_max()

        varianza = np.var(image)

        # Usar 'self.varianza_min' y 'self.varianza_max' ajustados globalmente
        radio_min = 5
        radio_max = 50

        # Normalizar la varianza para que esté entre 0 y 1
        varianza_norm = (varianza - self.varianza_min) / (self.varianza_max - self.varianza_min)
        varianza_norm = np.clip(varianza_norm, 0, 1)

        # Calcular el radio
        radio = radio_min + (radio_max - radio_min) * varianza_norm
        print(radio)
        return radio


img_rutas = {'top': 'imagenes/SENOS1-T.BMP', 'bottom': 'imagenes/SENOS1-B.BMP', 'left': 'imagenes/SENOS1-L.BMP',
             'right': 'imagenes/SENOS1-R.BMP', 'textura': 'imagenes/SENOS1-S.BMP'}
# plt.ion()
cargar = Cargarimagenes(img_rutas)
procesar = Procesarimagenes(cargar)

