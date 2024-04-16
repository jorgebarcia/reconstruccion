# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:55:10 2024

@author: Jorge
"""
import cv2
import matplotlib
matplotlib.use('TkAgg')
from scipy.integrate import cumulative_trapezoid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.linalg import lstsq
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

class Cargarimagenes:
    def __init__(self, img_rutas):
        # self.img_rutas = img_rutas
        self.img_dict = {}
        self.textura = None
        self.upload_img(img_rutas)

    def upload_img(self,img_rutas):  # self es nuestro objeto instancia, lo llamamos
        # en la funcion ya que img_ruta es un atributo de # nuestro objeto!!
        '''
        input: diccionario con nuestras rutas

        output: diccionario con las imagenes y la textura irá aparte
        ya que no tiene por que tener las mismas cualidades que las demas imagenes
        :return:
        '''
        for [key, ruta] in img_rutas.items():  # iteramos en ruta --> despues de este bucle en el nuevo diccionario
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
    def __init__(self, datos):
        self.datos= datos

        'atributos nuevos'
        self.ruido=None

        'aplicamos funciones'
        self.nivel_ruido()
        self.filtro(ver=False)
        self.aplicar_fourier(ver=False)

    def nivel_ruido(self):
        '''
        input: Nuestro objeto, mas en concreto el diccionario
        output: el ruido de nuestra imagen
        '''
        print("Valores Ruido")
        result_ruido={}
        for key, image in self.datos.img_dict.items():
            # if key != 'textura':  --> no hace falta, en esta nueva version no hay textura en img_dict
            self.ruido = np.std(image)
            print(f'El nivel de ruido de {key} es {self.ruido}')
            # return self.ruido #es necesario??'
            result_ruido[key] = self.ruido
        return result_ruido


    def filtro(self, sigma = 1, ver = True):
        '''
        Funcion que aplica un filtro gaussiano a nuestras imagenes
        Sabemos que tiene ruido gaussiano debido a los histogramas
        sigma: nivel de agresividad edl filtro
        '''
        for key, image in self.datos.img_dict.items():
            img_no_filtrada = self.datos.img_dict[key]
            self.datos.img_dict[key] = gaussian_filter(image, sigma=sigma)
            print(f'La imagen {key} se ha filtrado correctamente')

            if ver==True:
                canva=plt.figure(figsize=(8,3))
                original=canva.add_subplot(121)
                original.imshow(img_no_filtrada, cmap ='gray')
                original.set_title(f'Original {key}')
                original.axis('off')

                filtrada=canva.add_subplot(122)
                filtrada.imshow(self.datos.img_dict[key], cmap='gray')
                filtrada.set_title(f'Filtrada {key}')
                filtrada.axis('off')

                canva.tight_layout()
                plt.show(block=True)

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
        print('Valores Fourier')
        for key, image in self.datos.img_dict.items():
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

            self.datos.img_dict[key] = image_trans

    def calcular_varianzas(self):
        """
        Calcula la varianza de cada imagen en el diccionario img_dict y devuelve un diccionario
        de varianzas correspondiente a cada clave de imagen.
        """
        varianzas = {}
        for key, image in self.datos.img_dict.items():
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
        # solo si varianzas estan definidas
        if not hasattr(self, 'varianza_min') or not hasattr(self, 'varianza_max'):
            self.ajustar_varianza_min_max()

        varianza = np.var(image)
        radio_min = 5
        radio_max = 50

        # normalizams la varianza para que esté entre 0 y 1 (lo normal vaya)
        varianza_norm = (varianza - self.varianza_min) / (self.varianza_max - self.varianza_min)
        varianza_norm = np.clip(varianza_norm, 0, 1)

        radio = radio_min + (radio_max - radio_min) * varianza_norm
        print(f"vamos a usar un radio = {radio}")
        return radio

class Ecualizacion:
    def __init__(self,datos):
        "atributos"
        self.datos = datos

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
        print("Valores ecualizacion")
        for key, image in self.datos.img_dict.items():
            print(f"Procesando imagen {key}")

            contraste_antes = self.contraste(image)
            entropia_antes = self.entropia(image)

            # para ecualizar es necesario que las imagenes esten en formato de 8 bits
            # y si hacemos la transformada de fourier las convierte a 64 bits de coma flotante
            if image.dtype != np.uint8:
                print(f"la imagen {key} es una imagen {image.dtype}")
                image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                image = image.astype(np.uint8)

            # aplicamos ecualizacion CLAHE por que es mejor que .ecualhist()
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_ecualizada = clahe.apply(image)

             # Calcular contraste y entropía después de la ecualización
            contraste_despues = self.contraste(image_ecualizada)
            entropia_despues = self.entropia(image_ecualizada)

            # Actualizar la imagen en el diccionario
            self.datos.img_dict[key] = image_ecualizada

            # Imprimir mejoras
            print(f"Imagen {key} - Mejora de Contraste: {contraste_despues - contraste_antes}")
            print(f"Imagen {key} - Mejora de Entropía: {entropia_despues - entropia_antes}")

class Metodosaplanacion:
    def __init__(self,datos):
        self.datos = datos

        "funciones"
        self.aplicar_aplanacion()

    def aplanacion_poli(self, image):
        # Obtenemos dos matrices con índices cuya shape = imagen.shape
        index_y, index_x = np.indices(image.shape)

        # Aplanamos la imagen para operar sobre un array unidimensional
        image_flat = image.flatten()

        # Construimos la matriz de diseño para el ajuste polinómico
        # Por ejemplo, un polinomio de segundo grado tendría la forma:
        # z = c00 + c10*x + c01*y + c20*x^2 + c11*x*y + c02*y^2
        # Los coeficientes c10 y c01 corresponden al plano que ya has calculado.
        # Añadiríamos c20, c11, c02 para el ajuste polinómico de segundo grado.
        X_poly = np.column_stack((
            np.ones_like(image_flat),  # Término constante
            index_x.flatten(),  # x
            index_y.flatten(),  # y
            index_x.flatten() ** 2,  # x^2
            index_x.flatten() * index_y.flatten(),  # x*y
            index_y.flatten() ** 2  # y^2
        ))

        # Ajustamos el modelo
        coefs_poly, _, _, _ = lstsq(X_poly, image_flat)

        # Calculamos el plano polinómico
        plano_poly = X_poly @ coefs_poly.reshape(-1, 1)
        plano_poly = plano_poly.reshape(image.shape)

        # Restamos el plano de la imagen original
        image_corrected = image - plano_poly

        # Encuentra el valor mínimo y ajusta la imagen para que el mínimo sea 0
        min_value = np.min(image_corrected)
        image_corrected -= min_value

        # Opcionalmente, puedes reescalar la imagen a su rango original de intensidades
        image_corrected = (image_corrected / np.max(image_corrected)) * 255

        return image_corrected

    def aplicar_aplanacion(self):
        for key, image in self.datos.img_dict.items():
            self.datos.img_dict[key]=self.aplanacion(image)
            # self.datos.img_dict[key]=self.aplanacion_poli(image)

    def normalize_image(self,image, scale_min=0, scale_max=250):
        image_min = np.min(image)
        image_max = np.max(image)
        normalized_image = (image - image_min) / (image_max - image_min) * (scale_max - scale_min) + scale_min
        return normalized_image.astype(np.uint8)
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

        # min=np.min(image_correct)
        # print(min)
        # #
        # image_correct = image_correct - min
        # min=np.min(image_correct)
        # print(f'este es ahora{min}')

        num_cond = self.numero_condicion(matriz_sist, ver=False)

        residuo = self.evaluar_ajuste(image, coefi, matriz_sist)

        return image_correct
        # val_max=np.max(image)
        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 3, 1)
        # plt.imshow(image, cmap='gray')
        # plt.title('Imagen Original')
        # plt.text(f'Max: {val_max:.2f}',f'')
        # plt.colorbar()
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(plano, cmap='gray')
        # plt.title('Plano Substraído')
        # plt.colorbar()
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(image_correct, cmap='gray')
        # plt.title('Imagen Ajustada')
        # # plt.text('')
        # plt.colorbar()
        #
        # plt.show()
        # num_cond = self.numero_condicion(matriz_sist, ver=False)
        #
        # residuo = self.evaluar_ajuste(image, coefi, matriz_sist)
        #
        # return image_correct

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
    def __init__(self,datos):
        self.datos = datos
        self.dpixel = 1/251

        "funciones"
        self.integracion(1,1,0)
        # grad_x,grad_y=self.calcular_gradientes(1,1)
        # self.integracion_poisson(grad_x,grad_y)
        self.plot_superficie(ver_textura=True)
    def integracion(self, c, d, z0, eps=1e-5):
        # print(self.img_dict)
        # for key,image in self.img_dict.items():
        #     self.img_dict[key]=image.astype(np.float32)
        #     print(image.dtype)

        i_a = self.datos.img_dict['right'].astype(np.float32)
        # print(i_a.dtype)
        i_b = self.datos.img_dict['left'].astype(np.float32)
        i_c = self.datos.img_dict['top'].astype(np.float32)
        i_d = self.datos.img_dict['bottom'].astype(np.float32)

        # restriingimos la division por cero para uqe no nos salte ningun error
        s_dx = (i_a - i_b) / np.clip(i_a + i_b, eps, np.inf)
        s_dy = (i_c - i_d) / np.clip(i_c + i_d, eps, np.inf)

        # print(s_dx)
        # print(s_dy)

        # Acumulación a lo largo de axis=1 --> x / axis=0 -->y
        # z_x=cumtrapz(s_dx*c/d, dx=self.dpixel, axis=1, initial=z0)
        # z_y=cumtrapz(s_dy*c/d, dx=self.dpixel, axis=0, initial=z0)
        z_x = cumulative_trapezoid(s_dx * c / d, dx=self.dpixel, axis=1, initial=z0)
        z_y = cumulative_trapezoid(s_dy * c / d, dx=self.dpixel, axis=0, initial=z0)


        # z_x_reverse = np.fliplr(cumulative_trapezoid(np.fliplr(s_dx * c / d), dx=self.dpixel, axis=1, initial=z0))
        # z_y_reverse = np.flipud(cumulative_trapezoid(np.flipud(s_dy * c / d), dx=self.dpixel, axis=0, initial=z0))
        #
        # # Promedio de las integraciones en ambas direcciones
        # z_x_symmetric = (z_x + z_x_reverse) / 2
        # z_y_symmetric = (z_y + z_y_reverse) / 2
        #
        # # Combinación de las integraciones simétricas
        # self.z = z_x_symmetric + z_y_symmetric


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
        axis_1.plot_surface(x * self.dpixel, y * self.dpixel, self.z, cmap='plasma')

        axis_1.set_title('Topografia sin textura')
        axis_1.set_xlabel('X (mm)')
        axis_1.set_ylabel('Y (mm)')
        axis_1.set_zlabel('Z (altura en mm)')

        mappable = cm.ScalarMappable(cmap=cm.plasma)
        mappable.set_array(self.z)
        plt.colorbar(mappable, ax=axis_1, orientation='vertical', label='Altura (mm)', shrink=0.5, pad=0.2)

        if ver_textura and self.datos.textura is not None:

            con_textura = plt.figure()
            axis_2 = con_textura.add_subplot(111, projection='3d')

            if self.datos.textura.shape[0] != self.z.shape[0] or self.datos.textura.shape[1] != self.z.shape[1]:
                print(f'La forma de la imagen es: {self.datos.textura.shape}')
                print(f'La forma de la funcion es: {self.z.shape}')
                self.textura = cv2.resize(self.datos.textura, (self.z.shape[1], self.z.shape[0]))
                print(
                    'Hemos tenido que reajustar la dimension de la textura por que no coincidia, mira a ver que todo ande bien...')

            else:
                None

            axis_2.plot_surface(x * self.dpixel, y * self.dpixel, self.z, facecolors=self.datos.textura / 255.0, shade=False)

            axis_2.set_title('Topografia con textura')
            axis_2.set_xlabel('X (um)')
            axis_2.set_ylabel('Y (um)')
            axis_2.set_zlabel('Z (altura en um)')

            mappable_gray = cm.ScalarMappable(cmap=cm.gray)
            mappable_gray.set_array(self.datos.textura)
            plt.colorbar(mappable_gray, ax=axis_2, orientation='vertical', label='Intensidad', shrink=0.5, pad=0.2)

            plt.show()

    def calcular_gradientes(self, c, d, eps=1e-5):
        i_a = self.datos.img_dict['right'].astype(np.float32)
        i_b = self.datos.img_dict['left'].astype(np.float32)
        i_c = self.datos.img_dict['top'].astype(np.float32)
        i_d = self.datos.img_dict['bottom'].astype(np.float32)

        # Restringimos la división por cero para que no nos salte ningún error
        s_dx = (i_a - i_b) / np.clip(i_a + i_b, eps, np.inf)
        s_dy = (i_d - i_c) / np.clip(i_c + i_d, eps, np.inf)

        # Calculamos los gradientes ajustados por la constante c/d
        grad_x = s_dx * c / d
        grad_y = s_dy * c / d

        return grad_x, grad_y

    def integracion_poisson(self, grad_x, grad_y, frontera=None):
        ny, nx = grad_x.shape
        N = ny * nx

        # Coordenadas para los elementos no nulos de la matriz A
        row = np.array([])
        col = np.array([])
        data = np.array([])

        # Vector b
        b = np.zeros(N)

        for y in range(ny):
            for x in range(nx):
                i = x + y * nx

                # Asignamos el valor de la divergencia negativa de los gradientes a b
                b[i] = -grad_x[y, x] - grad_y[y, x]

                # Vecinos
                if x > 0:
                    row = np.append(row, i)
                    col = np.append(col, i - 1)
                    data = np.append(data, 1)
                if x < nx - 1:
                    row = np.append(row, i)
                    col = np.append(col, i + 1)
                    data = np.append(data, 1)
                if y > 0:
                    row = np.append(row, i)
                    col = np.append(col, i - nx)
                    data = np.append(data, 1)
                if y < ny - 1:
                    row = np.append(row, i)
                    col = np.append(col, i + nx)
                    data = np.append(data, 1)
                row = np.append(row, i)
                col = np.append(col, i)
                data = np.append(data, -4)

                # Condiciones de frontera de Dirichlet si 'frontera' es proporcionada
                if frontera is not None and (x == 0 or x == nx - 1 or y == 0 or y == ny - 1):
                    b[i] = frontera[y, x]

        # Construimos la matriz A usando formato COO para eficiencia
        A = coo_matrix((data, (row, col)), shape=(N, N))

        # Resolvemos el sistema lineal Ax = b
        x = spsolve(A.tocsr(), b)
        self.z=x.reshape((ny, nx))
        # Devolvemos la imagen reconstruida
        return self.z
class Contornos:
    def __init__(self, Reconstruccion):
        self.z = Reconstruccion.z
        self.dpixel = Reconstruccion.dpixel

        self.contornear_x(300)
        # self.muchos_contorno_x(5)

    def contornear_x(self, pos_y):
        # pos_y = 20  # 20 por ejemplo
        perfil = self.z[pos_y, :]
        perfil = gaussian_filter(perfil, sigma=1)
        ax_x = np.arange(len(perfil)) * self.dpixel

        'Ra'
        media_perfil = np.mean(perfil)

        Ra = np.mean(np.abs(
            perfil - media_perfil))  # rugosidad media aritmetica  -> promedio abs desviaciones a lo largo de la muestra
        Rmax = np.max(perfil)
        Rmin = np.min(perfil)

        'Rz'
        pikos, _ = find_peaks(perfil, distance=70, prominence=0.2)
        minimos, _ = find_peaks(-perfil, distance=95, prominence=0.2)

        pikos_val = perfil[pikos]  # valores nominales pikkos
        minimos_val = perfil[minimos]  # valores nominales minimos

        # np.argsort(pikos_alturas) --> nos devuelve un array =shape que tiene: [0]- mas bajo....[-1] mas alto
        # nos movemos en el espacio de indices de pikos_val
        pikos_5 = pikos[np.argsort(pikos_val)[-5:]]
        minimos_5 = minimos[np.argsort(minimos_val)[-5:]]

        # pasamos al espacio de indices de perfil a traves del orden hecho
        Rz = np.sum(np.abs(perfil[pikos_5])) / pikos_5.size + np.sum(
            np.abs(perfil[minimos_5])) / minimos_5.size  # por si acaso no hubiese 5 picos

        # ploteamos:
        contorno = plt.figure(figsize=(15, 5))
        ax = contorno.add_subplot(111)

        ax.plot(ax_x, perfil, '#4682B4', label=f'Contorno en y={pos_y}')
        ax.plot(ax_x[pikos_5], perfil[pikos_5], "x", color='red', label='Liberadores de Tensiones')
        ax.plot(ax_x[minimos_5], perfil[minimos_5], "x", color='k', label='Generadores de Tensiones')

        ax.hlines(Rmax, ax_x[0], ax_x[-1], '#FF4500', '--', label='Rmax')
        ax.hlines(Rmin, ax_x[0], ax_x[-1], '#FF4500', '--', label='Rmin')
        ax.hlines(media_perfil, ax_x[0], ax_x[-1], '#FFD700', '--', label='Media')
        ax.hlines(media_perfil + Ra, ax_x[0], ax_x[-1], 'g', '--', label='Desviacion estandar')
        ax.hlines(media_perfil - Ra, ax_x[0], ax_x[-1], 'g', '--')

        ax.text(ax_x[300], Rmax, f'{Rmax:.2f}', va='center', ha='right', backgroundcolor='w')
        ax.text(ax_x[300], Rmin, f'{Rmin:.2f}', va='center', ha='right', backgroundcolor='w')
        ax.text(ax_x[260], np.mean(perfil), f'{np.mean(perfil):.2f}', va='center', ha='right', backgroundcolor='w')

        # corchete Delta Z
        alto = 0.05
        ancho = ax_x[-1] - alto * 2
        ax.plot([ancho, ancho], [media_perfil, media_perfil + Ra], 'k-', lw=1)
        ax.plot([ancho - alto / 2, ancho + alto / 2], [media_perfil + Ra, media_perfil + Ra], 'k-', lw=1)
        ax.plot([ancho - alto / 2, ancho + alto / 2], [media_perfil, media_perfil], 'k-', lw=1)
        ax.text(ancho + alto, media_perfil + Ra / 2, f'Δz={Ra:.2f}', va='center', ha='left', backgroundcolor='w')
        print('se ha ejecutado no?')
        ax.legend()
        plt.show()

        def pilacontornos_x(self, ncontorno):
            pos_y = np.random.randint(0, self.z.shape[0], ncontorno)
            pos_y = np.sort(pos_y)

            # contornos_fig=plt.figure(figsize=(20, 10))
            contornos_fig = plt.figure()
            ax = contornos_fig.add_subplot(111, projection='3d')

            contorno_x = np.arange(self.z.shape[1]) * self.dpixel

            for i in pos_y:
                contorno = self.z[i, :]  # rugosidad concreta
                contorno = gaussian_filter(contorno, sigma=1)
                contorno_y = np.full_like(contorno_x, i * self.dpixel)
                ax.plot(contorno_x, contorno_y, contorno, label=f'Perfil en y={i}')
            ax.set_title('Perfiles de Rugosidad en 3D')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (altura en mm)')
            plt.legend()
            plt.show()

    def muchos_contorno_x(self, ncontorno):
        pos_y = np.random.randint(0, self.z.shape[0], ncontorno)
        pos_y = np.sort(pos_y)

        # contornos_fig=plt.figure(figsize=(20, 10))
        contornos_fig = plt.figure()
        ax = contornos_fig.add_subplot(111, projection='3d')

        contorno_x = np.arange(self.z.shape[1]) * self.dpixel

        for i in pos_y:
            contorno = self.z[i, :]  # rugosidad concreta
            contorno = gaussian_filter(contorno, sigma=1)
            contorno_y = np.full_like(contorno_x, i * self.dpixel)
            ax.plot(contorno_x, contorno_y, contorno, label=f'Perfil en y={i}')
        ax.set_title('Perfiles de Rugosidad en 3D')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (altura en mm)')
        plt.legend()
        plt.show()

class Histograma:
    def __init__(self,datos):
        self.datos = datos
        self.histogramear()
    def histogramear(self):

        histo_canva = plt.figure(figsize=(10, 20))

        for i, (key, image) in enumerate(self.datos.img_dict.items()):
            # if i<4:
            if key != 'textura':
                ax_img = histo_canva.add_subplot(4, 2, 2 * i + 1)
                imagen = ax_img.imshow(image, cmap='gray')
                ax_img.set_title(f'imagen {key}')
                ax_img.axis('off')

                ax_hist = histo_canva.add_subplot(4, 2, 2 * i + 2)
                hist = ax_hist.hist(image.ravel(), bins=256, range=[1, 256], color='gray', alpha=0.75)
                ax_hist.set_title(f'histograma imagen {key}')
            else:
                break

        # histo_canva.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.3)
        plt.show()


img_rutas = {'top': 'imagenes/SENOS1-T.BMP', 'bottom': 'imagenes/SENOS1-B.BMP', 'left': 'imagenes/SENOS1-L.BMP',
             'right': 'imagenes/SENOS1-R.BMP', 'textura': 'imagenes/SENOS1-S.BMP'}
# img_rutas = {'top': 'imagenes/CIRC1_T.BMP','bottom': 'imagenes/CIRC1_B.BMP','left': 'imagenes/CIRC1_L.BMP','right': 'imagenes/CIRC1_R.BMP','textura': 'imagenes/CIRC1.BMP'}

# img_rutas = {'top': 'imagenes/RUEDA1_T.BMP','bottom': 'imagenes/RUEDA1_B.BMP','left': 'imagenes/RUEDA1_L.BMP','right': 'imagenes/RUEDA1_R.BMP','textura': 'imagenes/RUEDA1_S.BMP'}
# img_rutas = {'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}


# plt.ion()

cargar = Cargarimagenes(img_rutas)
# histograma = Histograma(cargar)
# procesar = Procesarimagenes(cargar)
# ecualizar = Ecualizacion(cargar)
# histograma = Histograma(cargar)
# aplanar = Metodosaplanacion(cargar)
# procesar = Procesarimagenes(cargar)


reconstruir = Reconstruccion(cargar)
# contornear = Contornos(reconstruir)

