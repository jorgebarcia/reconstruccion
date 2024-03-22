# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:58:04 2024

@author: Jorge
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import cv2
import csv
from scipy.integrate import cumtrapz
from scipy.linalg import lstsq

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

from scipy.interpolate import RectBivariateSpline

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import matplotlib.pyplot as plt

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float
from scipy.interpolate import bisplrep, bisplev

class reco_superficie3d:
    def __init__(self,img_rutas,dpixel=10/251.8750):
        #le damos atributos de variables locales a nuestro objeto --> self
        
        self.img_ruta=img_rutas
        self.img_dict={}
        self.z=None # --> z es un atributo de nuestro objeto que calcularemos mas tarde, así nos curamos en salud con posibles errores si intentamos ver z antes de calcularlo
        self.dpixel=dpixel
        self.textura=None #lo mismo
        self.upload_imagenes()  # carga las imágenes
        # self.histogrameando()
        
        self.filtro_gaussiano()
        self.transformar()
        
        
        # self.aplanamoslocalmente()
        
        # self.filtro_gaussiano()
        # self.transformar()
        
        self.aplanacion()
        
        # self.aplanacionsk()
        
        # self.ecualizar()
        
        # self.filtro_gaussiano()
        self.transformar()
        # self.histogrameando()

    def upload_imagenes(self): #self es nuestro objeto, instancia, lo llamamos en la funcion ya que img_ruta es un atributo de nuestro objeto!!
        for key,ruta in self.img_ruta.items():  #key:nombre , image: ¡¡ojo!! son values e[0,255]
            
            if key=='textura':
                textura=cv2.imread(ruta,cv2.IMREAD_COLOR)
                if textura is None: #si imread no funciono, a textura le da el valor de None
                    print(f' La imagen de textura "{ruta}" no se pudo cargar. Verifica la ruta.')
                    continue  #salta  siguiente ítem del bucle si no se cargo bien la tal
                self.textura=cv2.cvtColor(textura, cv2.COLOR_BGR2RGB)
                    
            else:
                image=cv2.imread(ruta,cv2.IMREAD_GRAYSCALE)
                print(image.dtype)
                if image is None:
                    raise ValueError(f'La imagen {key} no se pudo cargar. Mira a ver que estén bien las rutas...')
   
            # self.img_dict[key]=image.astype(np.float32) #Vamos a necesitar coma flotante 32 bits para no perder informacion (cv.im... lee en 8 bits)
            self.img_dict[key]=image #si queremos ecualizar debemos usar esta linea en vez de la anterior
    
    'el otro dia'
    def aplicar_procesamiento_anisotropico_y_aplanamiento(self):
        for key, image in self.img_dict.items():
            if key != 'textura':
                # Primero aplicar suavizado anisotrópico
                image_suavizada = self.suavizar_anisotropico(image)
                # Después aplicar aplanamiento local
                image_aplanada = self.aplanamiento_local(image_suavizada)
                # Actualizar la imagen en el diccionario
                self.img_dict[key] = image_aplanada
    
    def suavizar_anisotropico(self, image):
        """
        Aplica suavizado anisotrópico a la imagen usando Non-Local Means Denoising.
        """
        # Convertir la imagen a float ya que NL Means requiere datos de flotante
        float_image = img_as_float(image)
        # Estimar el sigma del ruido de la imagen
        sigma_est = np.mean(estimate_sigma(float_image))
        # Aplicar Non-Local Means Denoising
        denoised_img = denoise_nl_means(float_image, h=1.15 * sigma_est, fast_mode=True,
                                        patch_size=5, patch_distance=3)
        return denoised_img

    def ajuste_segmentado(self, block_size=50, degree=2):
        nrows, ncols = self.z.shape
        image_corrected = np.zeros_like(self.z)

        for y in range(0, nrows, block_size):
            for x in range(0, ncols, block_size):
                # Asegúrate de no ir fuera de los límites de la imagen
                y_end = min(y + block_size, nrows)
                x_end = min(x + block_size, ncols)

                # Extrae el bloque
                block = self.z[y:y_end, x:x_end]
                y_coords, x_coords = np.mgrid[y:y_end, x:x_end]

                # Ajusta un polinomio o spline al bloque
                # Asumiendo que estás usando bisplrep y bisplev para hacer el ajuste spline
                # Bajo el supuesto de que bisplrep y bisplev son los métodos que deseas usar.
                # Nota: Debes asegurarte de que las entradas a bisplrep sean correctas y 
                # de que estés manejando los bordes de manera adecuada.
                spline_params = bisplrep(x_coords.ravel(), y_coords.ravel(), block.ravel(), kx=degree, ky=degree)
                block_fitted = bisplev(x_coords[:, 0], y_coords[0, :], spline_params).reshape((y_end-y, x_end-x))

                # Sustrae el modelo ajustado del bloque para aplanarlo
                block_corrected = block - block_fitted

                # Coloca el bloque aplanado en la imagen corregida
                image_corrected[y:y_end, x:x_end] = block_corrected

        self.z = image_corrected

    # def aplanamiento_local(self, image, block_size=50):
        """
        Aplana la imagen dividiéndola en bloques y ajustando un plano a cada uno.
        """
        nrows, ncols = image.shape
        image_corrected = np.copy(image)
        for y in range(0, nrows, block_size):
            for x in range(0, ncols, block_size):
                # Extraer bloque
                block = image[y:y+block_size, x:x+block_size]
                if block.size == 0:
                    
                    continue  # Saltar bloques vacíos por bordes irregulares
                # Aplanar bloque
                # block_corrected = self._aplanar_bloque(block)
                # block_corrected = self.aplanar(block)
                # block_corrected = self.aplanacionsk()
                block_corrected = self.ajuste_segmentado()
                # Colocar bloque corregido en la imagen
                image_corrected[y:y+block_size, x:x+block_size] = block_corrected
        return image_corrected
    
    # def _aplanar_bloque(self, block):
        """
        Aplana un bloque utilizando ajuste polinómico o mínimos cuadrados.
        """
        # Este es un lugar donde necesitarías implementar tu lógica de aplanamiento.
        # Puedes usar np.polyfit o alguna otra técnica de regresión para ajustar un
        # plano a los datos del bloque y luego restar este ajuste del bloque original.
        # Asegúrate de manejar los casos en que el bloque sea más pequeño que lo esperado
        # debido a los bordes de la imagen.
        
        # Ejemplo muy básico de ajuste y aplanamiento:
        nrows, ncols = block.shape
        y_indices, x_indices = np.indices(block.shape)
        
        # Ajuste polinómico de grado 1 (plano) a los datos del bloque
        coefs = np.polyfit(x_indices.ravel(), block.ravel(), 1)
        # Crear un plano usando los coeficientes del ajuste
        trend = np.polyval(coefs, x_indices)
        # Restar la tendencia del bloque para aplanar
        block_aplanado = block - trend
        
        return block_aplanado
    
    'hoy...'
    def aplanamoslocalmente(self):
        # Aplicar aplanamiento por bloques con dimensiones ajustadas a cada imagen
        for key, image in self.img_dict.items():
            if key != 'textura':
                self.img_dict[key] = self.aplanar_bloques_con_dimensiones_ajustadas(image)
    
    def aplanar_bloques_con_dimensiones_ajustadas(self, image):
        nrows, ncols = image.shape
        block_size_y = self.encontrar_factor_mas_cercano(nrows)
        block_size_x = self.encontrar_factor_mas_cercano(ncols)
        return self.aplanamiento_local(image, block_size_y, block_size_x)

    def encontrar_factor_mas_cercano(self, numero):
        for i in range(50, 0, -1):
            if numero % i == 0:
                return i
        return 1

    def aplanamiento_local(self, image, block_size_y, block_size_x):
        nrows, ncols = image.shape
        image_corrected = np.copy(image)
        for y in range(0, nrows, block_size_y):
            for x in range(0, ncols, block_size_x):
                block = image[y:y+block_size_y, x:x+block_size_x]
                if block.size == 0:
                    continue
                block_corrected = self._aplanar_bloque(block)
                image_corrected[y:y+block_size_y, x:x+block_size_x] = block_corrected
        return image_corrected
    
    def _aplanar_bloque(self, block):
        nrows, ncols = block.shape
        y, x = np.mgrid[:nrows, :ncols]
        X = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        y = block.flatten()
        
        # Crear un modelo polinómico de segundo grado
        model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        
        # Ajustar el modelo al bloque
        model.fit(X, y)
        
        # Predecir los valores de la superficie del polinomio
        z_poly = model.predict(X).reshape(nrows, ncols)
        
        # Calcular residuos
        residuos = block - z_poly
        
        # Calcular la norma de los residuos (suma de cuadrados de los errores)
        norma_residual = np.linalg.norm(residuos)
        
        # Calcular R^2
        r2 = r2_score(y, z_poly.flatten())
        
        # Puedes imprimir o almacenar los valores de R^2 y norma_residual para cada bloque
        print(f'Block R^2: {r2:.4f}')
        print(f'Block Residual Norm: {norma_residual:.4f}')
        
        # Devolver el bloque corregido sustrayendo la superficie ajustada
        block_corrected = block - z_poly
        return block_corrected
    
    
    # def aplanar_bloques_con_dimensiones_ajustadas(self):
    #     # Obtener las dimensiones de una imagen de ejemplo (assumimos que todas las imágenes tienen las mismas dimensiones)
    #     ejemplo_img = next(iter(self.img_dict.values()))
    #     nrows, ncols = ejemplo_img.shape

    #     # Ajustar el tamaño del bloque para que divida exactamente la imagen
    #     block_size_y = self.encontrar_factor_mas_cercano(nrows)
    #     block_size_x = self.encontrar_factor_mas_cercano(ncols)

    #     # Aplicar aplanamiento por bloques con las dimensiones ajustadas
    #     for key, image in self.img_dict.items():
    #         if key != 'textura':
    #             self.img_dict[key] = self.aplanamiento_local(image, block_size_y, block_size_x)

    # def encontrar_factor_mas_cercano(self, numero):
    #     # Encontrar el factor más grande que sea menor o igual que 50 y que divida exactamente al número
    #     for i in range(50, 0, -1):
    #         if numero % i == 0:
    #             return i
    #     return 1  # En caso de no encontrar un factor, regresar 1 (no es ideal)  --> como ocurra esto mi portatil explota xd

    # def aplanamiento_local(self, image, block_size_y, block_size_x):
    #     """
    #     Aplana la imagen dividiéndola en bloques y ajustando un plano a cada uno.
    #     """
    #     nrows, ncols = image.shape
    #     image_corrected = np.copy(image)
    #     for y in range(0, nrows, block_size_y):
    #         for x in range(0, ncols, block_size_x):
    #             # Extraer bloque
    #             block = image[y:y+block_size_y, x:x+block_size_x]
    #             if block.size == 0:
    #                 continue  # Saltar bloques vacíos por bordes irregulares
    #             # Aplanar bloque
    #             block_corrected = self._aplanar_bloque(block)
    #             # Colocar bloque corregido en la imagen
    #             image_corrected[y:y+block_size_y, x:x+block_size_x] = block_corrected

    #     return image_corrected

    # def _aplanar_bloque(self, block):
        
        
    #     """
    #     Aplana un bloque de la imagen ajustando un polinomio de segundo grado utilizando mínimos cuadrados.
    #     """
    #     # Obtener las coordenadas del bloque
    #     nrows, ncols = block.shape
    #     y, x = np.mgrid[:nrows, :ncols]

    #     # Preparar los datos para el ajuste
    #     X = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    #     y = block.flatten()

    #     # Crear un modelo polinómico de segundo grado
    #     model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

    #     # Ajustar el modelo al bloque
    #     model.fit(X, y)

    #     # Predecir los valores de la superficie del polinomio
    #     z_poly = model.predict(X).reshape(nrows, ncols)

    #     # Devolver el bloque corregido sustrayendo la superficie ajustada
    #     block_corrected = block - z_poly
    #     return block_corrected
                
    # def aplanamoslocalmente(self):
        for key, image in self.img_dict.items():
            if key !='textura':
                # self.img_dict[key] = self.aplanamiento_local(image)
                self.img_dict[key] = self.aplanar_bloques_con_dimensiones_ajustadas()
    
    'filtros'
    
    def filtro_gaussiano(self, sigma=1):
        # Aplicamos el filtro gaussiano a una copia de las imágenes para no alterar las originales
        img_dict_filtrado = {}
        for key, image in self.img_dict.items():
            if key != 'textura':  # Excluimos la imagen de textura del filtro
                img_filtrada = gaussian_filter(image, sigma=sigma)
                self.img_dict[key] = img_filtrada
                
                # Mostrar las imágenes originales y filtradas
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(image, cmap='gray')
                plt.title(f'Original: {key}')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(img_filtrada, cmap='gray')
                plt.title(f'Filtrada: {key}')
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
                
        # Actualizar el diccionario de imágenes con las versiones filtradas
        self.img_dict.update(img_dict_filtrado)

    def aplicar_transformada_fourier_y_evaluar(self,image):
        # Aplicar la FFT a la imagen
        f_transform = fft2(image)
        
        # Desplazar el cero de las frecuencias al centro
        f_shift = fftshift(f_transform)
        
        # Definir un radio para el filtro basado en el contenido de la imagen
        # Este valor puede ser ajustado o ser hecho variable como un parámetro
        r = 70
        # r=30
    
        # Crear un filtro pasa-bajas circular
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1
        
        # Aplicar la máscara y la transformada inversa de Fourier
        f_shift_masked = f_shift * mask
        f_ishift = ifftshift(f_shift_masked)
        image_back = ifft2(f_ishift)
        image_back = np.abs(image_back)
        
        # Comparar la imagen original con la filtrada
        ssim_value = ssim(image, image_back, data_range=image_back.max() - image_back.min())
        psnr_value = psnr(image, image_back, data_range=image_back.max() - image_back.min())
        
        print(f"SSIM: {ssim_value:.4f}")
        print(f"PSNR: {psnr_value:.4f}")
        
        # Visualizar
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(image_back, cmap='gray')
        plt.title('Filtrada')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # cuando mayor es el radio, menos suavizado, SSim y PSNR dan una buena idea de cuanto se desvia de la iamgen original
        return image_back, ssim_value, psnr_value
    
    def transformar(self):
        resultados = {}
        for key, image in self.img_dict.items():
            if key != 'textura':  # Si no quieres procesar la textura
                image_back, ssim_value, psnr_value = self.aplicar_transformada_fourier_y_evaluar(image)
                resultados[key] = {
                    'imagen_filtrada': image_back,
                    'ssim': ssim_value,
                    'psnr': psnr_value
                }
                # Aquí puedes decidir si quieres actualizar la imagen en el diccionario
                self.img_dict[key] = image_back
        return resultados
       
    'sklearn'
    def polinomio_aplanar(self, image, degree=3):
        # plt.ion()
        # Obtiene las coordenadas de los píxeles
        nrows, ncols = image.shape
        y, x = np.mgrid[:nrows, :ncols]
        # Aplanamos las coordenadas para hacerlas características
        X = np.vstack((x.ravel(), y.ravel())).T
        
        # Crea las características polinomiales
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # Crea y ajusta el modelo
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_poly, image.ravel())
        
        # Usamos el modelo para predecir los valores de z
        z_poly = model.predict(X_poly).reshape((nrows, ncols))
        
        # Calcula la imagen aplanada sustrayendo la superficie polinomial
        image_corrected = image - z_poly
        
        return image_corrected
                
    def evaluar_ajuste_polinomio(self, image, image_corrected):
        # Calcula los residuos
        residuos = image - image_corrected
        
        # Calcula la norma de los residuos
        norma_residual = np.linalg.norm(residuos)
        print(f"Norma de los residuos: {norma_residual}")

        # Calcula el coeficiente R²
        r2 = r2_score(image.ravel(), image_corrected.ravel())
        print(f"Coeficiente R²: {r2}")
        
        # Visualización
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(image_corrected, cmap='gray')
        plt.title('Ajustada')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(residuos, cmap='gray')
        plt.title('Residuos')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

        return norma_residual, r2
    
    def aplanacionsk(self):
        for key, image in self.img_dict.items():
            if key != 'textura':
                
                self.img_dict[key]=self.polinomio_aplanar(image)
                self.evaluar_ajuste_polinomio(image, self.img_dict[key])
    
    
    'plano normal'
    def obtener_numero_condicion(self, matriz_sist):
        numero_condicion = np.linalg.cond(matriz_sist)
        print(f"El número de condición de la matriz es: {numero_condicion}")
        
        # Visualización del número de condición en un gráfico
        plt.figure()
        plt.title("Número de Condición de la Matriz")
        plt.bar(['Matriz A'], [numero_condicion], color='blue')
        plt.ylabel('Número de Condición')
        plt.yscale('log')  # Escala logarítmica porque el número de condición puede ser muy grande
        plt.show()
        
        return numero_condicion
    
    def evaluar_ajuste_plano(self, image, coefi, matriz_sist):
        # Utiliza los coeficientes para predecir los valores de z
        z_predicho = matriz_sist @ coefi
        # Calcula los residuales (diferencia entre los valores reales y los predichos)
        residuales = image.ravel() - z_predicho
        
        # Calcula la norma (magnitud) de los residuales
        norma_residual = np.linalg.norm(residuales)
        print(f"La norma del residual es: {norma_residual}")
        
        # Evalúa si los residuales son pequeños o grandes
        if norma_residual < 1e-5:  # Puedes ajustar este umbral según lo que consideres 'pequeño'
            print("El ajuste es bueno.")
        else:
            print("El ajuste es potencialmente malo.")
    
        return norma_residual    
    
    'plano comun'
    def aplanacion(self):
        for key, image in self.img_dict.items():
            if key != 'textura':
                self.img_dict[key]=self.aplanar(image)
                
    
    def aplanar(self,image):
        '''
        Funcion que elimina las diferencias en las inclinaciones de las imagenes
        recogidas por los detectores BSE
        
        Emplearemos un algoritmo de minimos cuadrados -> deberemos aplanar nuestros arrays par aoperar sobre listas
        '''
        index_y,index_x=np.indices(image.shape) #obtenemos dos matrices con indices cuya shape=imagen.shape
        image_flat=image.flatten() #array.dim=1 (valores planos imagen)
        #ahora construimos matriz cada valor plano con sus indices (index_x,index_y,flat_value)
        
        matriz_sist=np.column_stack((index_x.flatten(),index_y.flatten(),np.ones_like(image_flat)))  #consume mas meemoria pero es mejor --> np.ones((imagen_flat.shape.. size))
        #z=c1*x+c2*y+c0, c0 es np.ones_like ya que son los valores de intensidad aplanados
        # A es una matriz que cada fila representa un punto x,y + un termino intependiente
        
        'realizamos el ajuste por minimos cuadrados'
        
        #mirar el condicionamiento de nuestraas matrices --> de aquí se puede sacar una grafica espectacular
        coefi,_,_,_=lstsq(matriz_sist,image_flat,lapack_driver='gelsy') # _ metodo para desechar variables... solo queremos llstsq[0]--> array.len=3 con coef c1,c2,c0
        # z=c1*x+c2*y+c0
        plano=(coefi[0]*index_x+coefi[1]*index_y+coefi[2]).reshape(image.shape)
        
        image_correct=image-plano
        
        num_cond = self.obtener_numero_condicion(matriz_sist)
        
        residuo = self.evaluar_ajuste_plano(image, coefi, matriz_sist)
        
        return image_correct
    
    def integracion(self,c,d,z0,eps=1e-5):
        
        # print(self.img_dict)
        # for key,image in self.img_dict.items():
        #     self.img_dict[key]=image.astype(np.float32)
        #     print(image.dtype)
        
        i_a=self.img_dict['right'].astype(np.float32)
        # print(i_a.dtype)
        i_b=self.img_dict['left'].astype(np.float32)
        i_c=self.img_dict['top'].astype(np.float32)
        i_d=self.img_dict['bottom'].astype(np.float32)
        
        #restriingimos la division por cero para uqe no nos salte ningun error
        s_dx=(i_a-i_b)/np.clip(i_a+i_b,eps,np.inf)
        s_dy=(i_d-i_c)/np.clip(i_c+i_d,eps,np.inf)
        
        # Acumulación a lo largo de axis=1 --> x / axis=0 -->y
        z_x=cumtrapz(s_dx*c/d, dx=self.dpixel, axis=1, initial=z0)
        z_y=cumtrapz(s_dy*c/d, dx=self.dpixel, axis=0, initial=z0)
        
        self.z=z_x+z_y #ahora self.z ya no es None
        print(np.max(self.z))
        print(np.min(self.z))
        
    def plot_superficie(self, ver_textura=True):     
        plt.ion()
        
        x,y=np.meshgrid(np.arange(self.z.shape[1]),np.arange(self.z.shape[0]))
        
        #primera figura
        sin_textura=plt.figure()
        axis_1=sin_textura.add_subplot(111,projection='3d')
        axis_1.plot_surface(x*self.dpixel, y*self.dpixel,self.z, cmap='viridis')
    
        axis_1.set_title('Topografia sin textura')
        axis_1.set_xlabel('X (mm)')
        axis_1.set_ylabel('Y (mm)')
        axis_1.set_zlabel('Z (altura en mm)')
        
        mappable = cm.ScalarMappable(cmap=cm.viridis)
        mappable.set_array(self.z)
        plt.colorbar(mappable, ax=axis_1, orientation='vertical', label='Altura (mm)',shrink=0.5,pad=0.2)
        
        if ver_textura and self.textura is not None:
            
            con_textura=plt.figure()
            axis_2=con_textura.add_subplot(111,projection='3d')
            
            if self.textura.shape[0] !=self.z.shape[0] or self.textura.shape[1] !=self.z.shape[1]:
                print(f'La forma de la imagen es: {self.textura.shape}')
                print(f'La forma de la funcion es: {self.z.shape}')
                self.textura=cv2.resize(self.textura,(self.z.shape[1],self.z.shape[0]))
                print('Hemos tenido que reajustar la dimension de la textura por que no coincidia, mira a ver que todo ande bien...')
                
            else: None

            axis_2.plot_surface(x*self.dpixel, y*self.dpixel, self.z, facecolors=self.textura/255.0, shade=False)
         
            axis_2.set_title('Topografia con textura')
            axis_2.set_xlabel('X (um)')
            axis_2.set_ylabel('Y (um)')
            axis_2.set_zlabel('Z (altura en um)')
            
            mappable_gray = cm.ScalarMappable(cmap=cm.gray)
            mappable_gray.set_array(self.textura)
            plt.colorbar(mappable_gray, ax=axis_2, orientation='vertical', label='Intensidad', shrink=0.5, pad=0.2)
            
            plt.show()
            
    def contornear_x(self,pos_y):
        
        pos_y=20  #20 por ejemplo
        perfil=self.z[pos_y, :]
        perfil=gaussian_filter(perfil, sigma=1)
        ax_x=np.arange(len(perfil))*self.dpixel
        
        'Ra'
        media_perfil=np.mean(perfil)
        
        Ra=np.mean(np.abs(perfil-media_perfil)) #rugosidad media aritmetica  -> promedio abs desviaciones a lo largo de la muestra
        Rmax=np.max(perfil)  
        Rmin=np.min(perfil)
        
        'Rz'
        pikos,_=find_peaks(perfil, distance=70, prominence=0.2)
        minimos,_=find_peaks(-perfil, distance=95, prominence=0.2)

        pikos_val=perfil[pikos] #valores nominales pikkos
        minimos_val=perfil[minimos] #valores nominales minimos
        
        #np.argsort(pikos_alturas) --> nos devuelve un array =shape que tiene: [0]- mas bajo....[-1] mas alto
        #nos movemos en el espacio de indices de pikos_val
        pikos_5 = pikos[np.argsort(pikos_val)[-5:]] 
        minimos_5 = minimos[np.argsort(minimos_val)[-5:]]
        
        #pasamos al espacio de indices de perfil a traves del orden hecho
        Rz = np.sum(np.abs(perfil[pikos_5]))/pikos_5.size + np.sum(np.abs(perfil[minimos_5])) / minimos_5.size #por si acaso no hubiese 5 picos
        
        #ploteamos:
        contorno=plt.figure(figsize=(15,5))
        ax=contorno.add_subplot(111)
        
        ax.plot(ax_x,perfil,'#4682B4',label=f'Contorno en y={pos_y}')
        ax.plot(ax_x[pikos_5], perfil[pikos_5], "x", color='red', label='Liberadores de Tensiones')
        ax.plot(ax_x[minimos_5], perfil[minimos_5], "x", color='k', label='Generadores de Tensiones')

        ax.hlines(Rmax,ax_x[0],ax_x[-1],'#FF4500','--',label='Rmax')
        ax.hlines(Rmin,ax_x[0],ax_x[-1],'#FF4500','--', label='Rmin')    
        ax.hlines(media_perfil,ax_x[0],ax_x[-1],'#FFD700','--', label='Media')    
        ax.hlines(media_perfil+Ra, ax_x[0],ax_x[-1],'g', '--', label='Desviacion estandar')
        ax.hlines(media_perfil-Ra,ax_x[0],ax_x[-1],'g','--')
        
        ax.text(ax_x[300], Rmax, f'{Rmax:.2f}', va='center', ha='right', backgroundcolor='w')
        ax.text(ax_x[300], Rmin, f'{Rmin:.2f}', va='center', ha='right', backgroundcolor='w')
        ax.text(ax_x[260], np.mean(perfil), f'{np.mean(perfil):.2f}', va='center', ha='right', backgroundcolor='w')
        
        #corchete Delta Z
        alto=0.05
        ancho=ax_x[-1]-alto*2
        ax.plot([ancho,ancho], [media_perfil,media_perfil+Ra],'k-',lw=1)
        ax.plot([ancho-alto/2, ancho+alto/2], [media_perfil+Ra, media_perfil+Ra], 'k-', lw=1)
        ax.plot([ancho-alto/2, ancho+alto/2], [media_perfil, media_perfil], 'k-', lw=1)
        ax.text(ancho+alto, media_perfil+Ra/2, f'Δz={Ra:.2f}',va='center', ha='left', backgroundcolor='w')
        
        ax.legend()
        plt.show()
        
    def pilacontornos_x(self,ncontorno):
        pos_y=np.random.randint(0,self.z.shape[0],ncontorno)
        pos_y=np.sort(pos_y)

        # contornos_fig=plt.figure(figsize=(20, 10))
        contornos_fig=plt.figure()
        ax=contornos_fig.add_subplot(111, projection='3d')
        
        contorno_x=np.arange(self.z.shape[1])*self.dpixel 
        
        for i in pos_y:
            contorno= self.z[i,:] #rugosidad concreta
            contorno= gaussian_filter(contorno, sigma=1)
            contorno_y= np.full_like(contorno_x,i*self.dpixel) 
            ax.plot(contorno_x, contorno_y,contorno, label=f'Perfil en y={i}')        
        ax.set_title('Perfiles de Rugosidad en 3D')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (altura en mm)')
        plt.legend()
        plt.show()
        
    def histogrameando(self):

        histo_canva=plt.figure(figsize=(10,20))

        for i, (key,image) in enumerate(self.img_dict.items()):
            # if i<4:      
            if key != 'textura':                  
                ax_img=histo_canva.add_subplot(4,2,2*i+1)
                imagen=ax_img.imshow(image,cmap='gray')
                ax_img.set_title(f'imagen {key}')
                ax_img.axis('off')
                
                ax_hist=histo_canva.add_subplot(4,2,2*i+2)
                hist=ax_hist.hist(image.ravel(),bins=256,range=[1,256],color='gray',alpha=0.75)
                ax_hist.set_title(f'histograma imagen {key}')
            else: break
        
        # histo_canva.tight_layout()
        plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1,hspace=0.4,wspace=0.3)
        plt.show()           
    
    def ecualizar(self):
        for key, image in self.img_dict.items():
            #imagen en escala gris?
            if image.ndim == 2 or image.shape[2] == 1:
                print(image.dtype)
                # self.img_dict[key] = cv2.equalizeHist(image)
                #cambiara si hago esto¿?
                if image.dtype != np.uint8:
                    print(image.dtype)
                    print('algo raro hayyy eh')
                    
                    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    image = image.astype(np.uint8)
                
                # Ecualizar la imagen
                self.img_dict[key] = cv2.equalizeHist(image)
            else:
                print(f"La imagen {key} no ta en escala de grises y no se puede ecuañlizar.")

# plt.ion()

img_rutas = {'top': 'SENOS1-T.BMP','bottom': 'SENOS1-B.BMP','left': 'SENOS1-L.BMP','right': 'SENOS1-R.BMP','textura': 'SENOS1-S.BMP'}
# img_rutas = {'top': 'CIRC1_T.BMP','bottom': 'CIRC1_B.BMP','left': 'CIRC1_L.BMP','right': 'CIRC1_R.BMP','textura': 'CIRC1.BMP'}

# img_rutas = {'top': 'RUEDA1_T.BMP','bottom': 'RUEDA1_B.BMP','left': 'RUEDA1_L.BMP','right': 'RUEDA1_R.BMP','textura': 'RUEDA1_S.BMP'}
# img_rutas = {'top': 'RUEDA3_T.BMP','bottom': 'RUEDA3_B.BMP','left': 'RUEDA3_L.BMP','right': 'RUEDA3_R.BMP','textura': 'RUEDA3.BMP'}

mi_superficie = reco_superficie3d(img_rutas)

mi_superficie.integracion(c=1, d=1, z0=0)


mi_superficie.plot_superficie(ver_textura=True)

# mi_superficie.contornear_x(20)
    