# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:55:10 2024

@author: Jorge
"""
import cv2
import matplotlib
from scipy.integrate import cumulative_trapezoid
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
# from scipy.signal import find_peaks

# from scipy.integrate import cumulative_trapezoid
from scipy.linalg import lstsq

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

class Ecualizacion:
    def __init__(self,atributos_procesar):
        "atributos"
        self.img_dict=atributos_procesar.img_dict

        "funciones"
        self.ecualizar()
    def contraste(self,image):
        return np.std(image)

    def entropia(self,image):
        hist,_=np.histogram(image.flatten(),bins=256,range=(0,256))
        hist_norm=hist/hist.sum() #normalizamos el histograma
        # vamos a calcular la entropia
        S = -np.sum(hist_norm * np.log2(hist_norm + np.finfo(float).eps)) #con np.finfo evitamos un posible log 0
        return  S


    def ecualizar(self):
        for key, image in self.img_dict.items():
            print(f"Procesando imagen {key}")

            contraste_antes = self.contraste(image)
            entropia_antes = self.entropia(image)

            # para ecualizar es necesario que las imagenes esten en formato de 8 bits
            # y si hacemos la transformada de fourier las convierte a 64 bits de coma flotante
            if image.dtype != np.uint8:
                print(f"la imagen {key} es una imagen {image.dtype}")
                image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                image = image.astype(np.uint8)

            # Aplicar CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_ecualizada = clahe.apply(image)

             # Calcular contraste y entropía después de la ecualización
            contraste_despues = self.contraste(image_ecualizada)
            entropia_despues = self.entropia(image_ecualizada)

            # Actualizar la imagen en el diccionario
            self.img_dict[key] = image_ecualizada

            # Imprimir mejoras
            print(f"Imagen {key} - Mejora de Contraste: {contraste_despues - contraste_antes}")
            print(f"Imagen {key} - Mejora de Entropía: {entropia_despues - entropia_antes}")

class Metodosaplanacion:
    def __init__(self,atributos_ecualizacion):
        self.img_dict=atributos_ecualizacion.img_dict

        "funciones"
        self.aplicar_aplanacion()

    def aplicar_aplanacion(self):
        for key, image in self.img_dict.items():
            self.img_dict[key]=self.aplanacion(image)
    def aplanacion(self, image):
        '''
        Aplanacion por ajuste mediante MINIMOS CUADRADOS!!

        Funcion que elimina las diferencias en las inclinaciones de las imagenes
        recogidas por los detectores BSE

        Emplearemos un algoritmo de minimos cuadrados -> deberemos aplanar nuestros arrays par aoperar sobre listas
        '''
        index_y, index_x = np.indices(image.shape)  # obtenemos dos matrices con indices cuya shape=imagen.shape
        image_flat = image.flatten()  # array.dim=1 (valores planos imagen)
        # ahora construimos matriz cada valor plano con sus indices (index_x,index_y,flat_value)

        matriz_sist = np.column_stack((index_x.flatten(), index_y.flatten(), np.ones_like(
            image_flat)))  # consume mas meemoria pero es mejor --> np.ones((imagen_flat.shape.. size))
        # z=c1*x+c2*y+c0, c0 es np.ones_like ya que son los valores de intensidad aplanados
        # A es una matriz que cada fila representa un punto x,y + un termino intependiente
        'realizamos el ajuste por minimos cuadrados'

        # mirar el condicionamiento de nuestraas matrices --> de aquí se puede sacar una grafica espectacular
        coefi, _, _, _ = lstsq(matriz_sist, image_flat,
                               lapack_driver='gelsy')  # _ metodo para desechar variables... solo queremos llstsq[0]--> array.len=3 con coef c1,c2,c0

        # z=c1*x+c2*y+c0
        plano = (coefi[0] * index_x + coefi[1] * index_y + coefi[2]).reshape(image.shape)

        image_correct = image - plano

        num_cond = self.numero_condicion(matriz_sist, ver=False)

        residuo = self.evaluar_ajuste(image, coefi, matriz_sist)

        return image_correct

    def numero_condicion(self, matriz_sist, ver=True):
        numero_condicion = np.linalg.cond(matriz_sist)
        print(f"El número de condición de la matriz es: {numero_condicion}")
        if ver == True:
            # Visualización del número de condición en un gráfico
            plt.figure()
            plt.title("Número de Condición de la Matriz")
            plt.bar(['Matriz A'], [numero_condicion], color='blue')
            plt.ylabel('Número de Condición')
            plt.yscale('log')  # Escala logarítmica porque el número de condición puede ser muy grande
            plt.show()

        return numero_condicion

    def evaluar_ajuste(self, image, coefi, matriz_sist):
        # usamos los coeficientes para predecir los valores de z
        z_predicho = matriz_sist @ coefi

        # residuales (diferencia entre los valores reales y los predichos)
        residuales = image.ravel() - z_predicho

        # Calcula la norma de los residuales
        norma_residual = np.linalg.norm(residuales)
        print(f"La norma del residual es: {norma_residual}")


        if norma_residual < 1e-5:
            print("El ajuste es bueno.")
        else:
            print("El ajuste es potencialmente malo.")

        return norma_residual



class Reconstruccion:
    def __init__(self,atributos_cargar,atributos_ecualizar):
        self.img_dict = atributos_ecualizar.img_dict
        # self.img_dict = atributos_aplanar.img_dict
        self.dpixel = 1/251
        self.textura = atributos_cargar.textura
        "funciones"
        self.integracion(1,1,0)
        self.plot_superficie(ver_textura=True)
    def integracion(self, c, d, z0, eps=1e-5):
        # print(self.img_dict)
        # for key,image in self.img_dict.items():
        #     self.img_dict[key]=image.astype(np.float32)
        #     print(image.dtype)

        i_a = self.img_dict['right'].astype(np.float32)
        # print(i_a.dtype)
        i_b = self.img_dict['left'].astype(np.float32)
        i_c = self.img_dict['top'].astype(np.float32)
        i_d = self.img_dict['bottom'].astype(np.float32)

        # restriingimos la division por cero para uqe no nos salte ningun error
        s_dx = (i_a - i_b) / np.clip(i_a + i_b, eps, np.inf)
        s_dy = (i_d - i_c) / np.clip(i_c + i_d, eps, np.inf)

        # print(s_dx)
        # print(s_dy)

        # Acumulación a lo largo de axis=1 --> x / axis=0 -->y
        # z_x=cumtrapz(s_dx*c/d, dx=self.dpixel, axis=1, initial=z0)
        # z_y=cumtrapz(s_dy*c/d, dx=self.dpixel, axis=0, initial=z0)
        z_x = cumulative_trapezoid(s_dx * c / d, dx=self.dpixel, axis=1, initial=z0)
        z_y = cumulative_trapezoid(s_dy * c / d, dx=self.dpixel, axis=0, initial=z0)

        self.z = z_x + z_y  # ahora self.z ya no es None
        # print(self.z)
        # print(np.max(self.z))
        # print(np.min(self.z))
        # self.z=self.z/(np.max(self.z))
        # print(np.max(self.z))
        # print(np.min(self.z))

    def plot_superficie(self, ver_textura=True):
        # plt.ion()

        x, y = np.meshgrid(np.arange(self.z.shape[1]), np.arange(self.z.shape[0]))

        # primera figura
        sin_textura = plt.figure()
        axis_1 = sin_textura.add_subplot(111, projection='3d')
        axis_1.plot_surface(x * self.dpixel, y * self.dpixel, self.z, cmap='viridis')

        axis_1.set_title('Topografia sin textura')
        axis_1.set_xlabel('X (mm)')
        axis_1.set_ylabel('Y (mm)')
        axis_1.set_zlabel('Z (altura en mm)')

        mappable = cm.ScalarMappable(cmap=cm.viridis)
        mappable.set_array(self.z)
        plt.colorbar(mappable, ax=axis_1, orientation='vertical', label='Altura (mm)', shrink=0.5, pad=0.2)

        if ver_textura and self.textura is not None:

            con_textura = plt.figure()
            axis_2 = con_textura.add_subplot(111, projection='3d')

            if self.textura.shape[0] != self.z.shape[0] or self.textura.shape[1] != self.z.shape[1]:
                print(f'La forma de la imagen es: {self.textura.shape}')
                print(f'La forma de la funcion es: {self.z.shape}')
                self.textura = cv2.resize(self.textura, (self.z.shape[1], self.z.shape[0]))
                print(
                    'Hemos tenido que reajustar la dimension de la textura por que no coincidia, mira a ver que todo ande bien...')

            else:
                None

            axis_2.plot_surface(x * self.dpixel, y * self.dpixel, self.z, facecolors=self.textura / 255.0, shade=False)

            axis_2.set_title('Topografia con textura')
            axis_2.set_xlabel('X (um)')
            axis_2.set_ylabel('Y (um)')
            axis_2.set_zlabel('Z (altura en um)')

            mappable_gray = cm.ScalarMappable(cmap=cm.gray)
            mappable_gray.set_array(self.textura)
            plt.colorbar(mappable_gray, ax=axis_2, orientation='vertical', label='Intensidad', shrink=0.5, pad=0.2)

            plt.show()

# img_rutas = {'top': 'imagenes/SENOS1-T.BMP', 'bottom': 'imagenes/SENOS1-B.BMP', 'left': 'imagenes/SENOS1-L.BMP',
             'right': 'imagenes/SENOS1-R.BMP', 'textura': 'imagenes/SENOS1-S.BMP'}

# img_rutas = {'top': 'CIRC1_T.BMP','bottom': 'CIRC1_B.BMP','left': 'CIRC1_L.BMP','right': 'CIRC1_R.BMP','textura': 'CIRC1.BMP'}

img_rutas = {'top': 'imagenes/RUEDA1_T.BMP','bottom': 'imagenes/RUEDA1_B.BMP','left': 'imagenes/RUEDA1_L.BMP','right': 'imagenes/RUEDA1_R.BMP','textura': 'imagenes/RUEDA1_S.BMP'}
# img_rutas = {'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}


# plt.ion()
cargar = Cargarimagenes(img_rutas)
procesar = Procesarimagenes(cargar)

ecualizar = Ecualizacion(procesar)
# ecualizar = Reconstruccion(cargar)
# aplanar = Metodosaplanacion(ecualizar)
integrar_y_plotear = Reconstruccion(cargar,ecualizar)