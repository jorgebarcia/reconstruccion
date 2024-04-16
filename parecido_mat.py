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
from scipy import linalg
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
        # self.nivel_ruido()
        self.filtro(ver=False)
        # self.aplicar_fourier(ver=False)
        # self.filtro(ver=False)

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


    def filtro(self, sigma = 20, ver = True):
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
            # r=200
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

    def aplicar_aplanacion(self):
        for key, image in self.datos.img_dict.items():
            self.datos.img_dict[key]=self.aplanacion(image)

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


        # val_max=np.max(image)
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 3, 1)
        # plt.imshow(image, cmap='gray')
        # plt.title(f'Imagen Original - Max: {val_max}')
        # plt.colorbar()
        #
        # val_max = np.max(plano)
        # plt.subplot(1, 3, 2)
        # plt.imshow(plano, cmap='gray')
        # plt.title(f'Plano - Max: {val_max}')
        # plt.colorbar()
        #
        # val_max= np.max(image_correct)
        # plt.subplot(1, 3, 3)
        # plt.imshow(image_correct, cmap='gray')
        # plt.title(f'Imagen Ajustada - Max: {val_max}')
        # # plt.text('')
        # plt.colorbar()
        # plt.show()


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
    def __init__(self,datos):
        self.datos = datos
        self.dpixel = 500/251

        "funciones"
        # self.integracion(1,1,0)
        self.integrar_bidireccional(1,1,0)
        # self.integrar_bidireccional_python()

        # self.corregir_plano()
        # self.corregir_polinomio()
        self.plot_superficie(ver_textura=True)
    def integracion(self, c, d, z0, eps=1e-5):
        # print(self.img_dict)

        i_a = self.datos.img_dict['right'].astype(np.float32)
        # print(i_a.dtype)
        i_b = self.datos.img_dict['left'].astype(np.float32)
        i_c = self.datos.img_dict['top'].astype(np.float32)
        i_d = self.datos.img_dict['bottom'].astype(np.float32)

        # restriingimos la division por cero para uqe no nos salte ningun error
        s_dx = (i_a - i_b) / np.clip(i_a + i_b, eps, np.inf)
        s_dy = (i_d - i_c) / np.clip(i_c + i_d, eps, np.inf)


        z_x = cumulative_trapezoid(s_dx * c / d, dx=self.dpixel, axis=1, initial=z0)
        z_y = cumulative_trapezoid(s_dy * c / d, dx=self.dpixel, axis=0, initial=z0)

        self.z = z_x + z_y  # ahora self.z ya no es None

        media = self.z.mean()
        desviacion = self.z.std()
        print(media, desviacion)

    def integrar_bidireccional(self,c,d,z0, eps=1e-5):
        i_a = self.datos.img_dict['right'].astype(np.float32)
        # print(i_a.dtype)
        i_b = self.datos.img_dict['left'].astype(np.float32)
        i_c = self.datos.img_dict['top'].astype(np.float32)
        i_d = self.datos.img_dict['bottom'].astype(np.float32)

        # restriingimos la division por cero para uqe no nos salte ningun error
        s_dx = (i_a - i_b) / np.clip(i_a + i_b, eps, np.inf)
        s_dy = (i_d - i_c) / np.clip(i_c + i_d, eps, np.inf)

        # Integración de izquierda a derecha en x
        z_lr = np.cumsum(s_dx * self.dpixel, axis=1)
        # Integración de derecha a izquierda en x (invierte la matriz, integra, y luego invierte el resultado)
        z_rl = np.cumsum(np.flip(s_dx, axis=1) * self.dpixel, axis=1)
        z_rl = np.flip(z_rl, axis=1)

        # Integración de arriba a abajo en y
        z_tb = np.cumsum(s_dy * self.dpixel, axis=0)
        # Integración de abajo hacia arriba en y (invierte la matriz, integra, y luego invierte el resultado)
        z_bt = np.cumsum(np.flip(s_dy, axis=0) * self.dpixel, axis=0)
        z_bt = np.flip(z_bt, axis=0)

        # Combinación de las integraciones
        z_combined = (z_lr + z_rl + z_tb + z_bt) / 4
        self.z=z_combined
        media= z_combined.mean()
        desviacion=z_combined.std()
        print(f'la media es {media}')
        print(f'la desviacion es {desviacion}')

        return self.z

    def integrar_bidireccional_python(self):
        """
        images: diccionario con las imágenes 'top', 'bottom', 'left', 'right'
        dpixel: tamaño de pixel en las unidades deseadas
        """
        # Convertimos las imágenes a flotantes y normalizamos si es necesario
        i_top = self.datos.img_dict['top'].astype(np.float32)
        i_bottom = self.datos.img_dict['bottom'].astype(np.float32)
        i_left = self.datos.img_dict['left'].astype(np.float32)
        i_right = self.datos.img_dict['right'].astype(np.float32)

        # Calculamos las diferencias normalizadas (gradientes)
        s_dx = (i_right - i_left) / (i_right + i_left + np.spacing(1))
        s_dy = (i_bottom - i_top) / (i_bottom + i_top + np.spacing(1))

        # Integramos las gradientes
        z_lr = np.cumsum(s_dx, axis=1) * self.dpixel
        z_rl = np.cumsum(np.fliplr(s_dx), axis=1) * self.dpixel
        z_rl = np.fliplr(z_rl)

        z_tb = np.cumsum(s_dy, axis=0) * self.dpixel
        z_bt = np.cumsum(np.flipud(s_dy), axis=0) * self.dpixel
        z_bt = np.flipud(z_bt)

        # Promediamos las superficies integradas en cada dirección
        z_combined = (z_lr + np.fliplr(z_rl) + z_tb + np.flipud(z_bt)) / 4

        # Corrección final: restar la media para centrar la superficie
        z_corrected = z_combined - np.mean(z_combined)
        self.z = z_combined
        # return z_corrected

    def corregir_plano(self):
        z = self.z
        # Creamos una malla de coordenadas basada en la topografía
        x_index, y_index = np.indices(z.shape)

        # Preparamos la matriz de diseño para la regresión lineal múltiple, incluyendo un término constante
        X = np.stack((x_index.ravel(), y_index.ravel(), np.ones_like(x_index).ravel()), axis=-1)

        # Vector de topografía (variable respuesta)
        Y = z.ravel()

        # Ajustamos el modelo lineal (plano) a los datos
        coeficientes, residuals, rank, s = linalg.lstsq(X, Y)

        # Calculamos el plano estimado usando los coeficientes obtenidos
        plano_estimado = X @ coeficientes
        plano_estimado = plano_estimado.reshape(z.shape)

        # Restamos el plano estimado de la topografía original para corregir la inclinación
        z_corregido = z - plano_estimado
        self.z = z_corregido

        # Mostrar resultados
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Original Topography
        cax_0 = axs[0].imshow(z, cmap='plasma')
        fig.colorbar(cax_0, ax=axs[0], orientation='vertical')
        axs[0].set_title('Topografía Original')
        axs[0].axis('off')

        # Plano Estimado
        cax_1 = axs[1].imshow(plano_estimado, cmap='plasma')
        fig.colorbar(cax_1, ax=axs[1], orientation='vertical')
        axs[1].set_title('Plano Estimado')
        axs[1].axis('off')

        # Topografía Corregida
        cax_2 = axs[2].imshow(z_corregido, cmap='plasma')
        fig.colorbar(cax_2, ax=axs[2], orientation='vertical')
        axs[2].set_title('Topografía Corregida')
        axs[2].axis('off')

        plt.show()

        # Imprimir métricas
        print("Coeficientes del plano:", coeficientes)
        print("Suma de residuos cuadrados:", residuals)

        # Calcular métricas adicionales
        mse = np.mean((z - z_corregido) ** 2)
        mae = np.mean(np.abs(z - z_corregido))
        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)

        # return z_corregido, plano_estimado, residuals, coeficientes

    def corregir_polinomio(self, grado=2):
        z = self.z
        x_index, y_index = np.indices(z.shape)

        # Generar los términos del polinomio
        X = np.ones(z.shape).ravel()
        for i in range(1, grado + 1):
            for j in range(i + 1):
                X = np.vstack((X, (x_index ** (i - j) * y_index ** j).ravel()))

        # Vector de topografía (variable respuesta)
        Y = z.ravel()

        # Ajustar el modelo polinomial a los datos
        coeficientes, residuals, rank, s = linalg.lstsq(X.T, Y)

        # Calcular la superficie polinomial estimada usando los coeficientes obtenidos
        z_estimado = np.dot(X.T, coeficientes).reshape(z.shape)

        # Calcular la topografía corregida
        z_corregido = z - z_estimado

        # Calcular errores
        mse = np.mean((z - z_estimado) ** 2)
        mae = np.mean(np.abs(z - z_estimado))

        # Visualización
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # cmaps = ['viridis', 'plasma', 'inferno']
        # cmaps=['plasma']

        for ax, data, title in zip(axs, [z, z_estimado, z_corregido],
                                         ['Topografía Original', 'Superficie Polinomial Estimada',
                                          'Topografía Corregida'],
                                   ):
            img = ax.imshow(data, cmap='plasma')
            ax.set_title(title)
            ax.axis('off')
            fig.colorbar(img, ax=ax)

        plt.show()

        # Métricas
        print("Coeficientes del polinomio:", coeficientes)
        print("Suma de residuos cuadrados:", residuals)
        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)
        self.z=z_corregido
        # return z_corregido, z_estimado, coeficientes, mse, mae


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
            # axis_2.set_zlim(-20,1000)

            mappable_gray = cm.ScalarMappable(cmap=cm.gray)
            mappable_gray.set_array(self.datos.textura)
            plt.colorbar(mappable_gray, ax=axis_2, orientation='vertical', label='Intensidad', shrink=0.5, pad=0.2)

            plt.show()

class Contornos:
    def __init__(self, Reconstruccion):
        self.z = Reconstruccion.z
        self.dpixel = Reconstruccion.dpixel

        self.contornear_x(300)
        self.contornear_y(300)
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
        contorno = plt.figure(figsize=(12,4))
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

    def contornear_y(self, pos_x):
        # pos_y = 20  # 20 por ejemplo
        perfil = self.z[:, pos_x]
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
        contorno = plt.figure(figsize=(12, 4))
        ax = contorno.add_subplot(111)

        ax.plot(ax_x, perfil, '#4682B4', label=f'Contorno en y={pos_x}')
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
        stats = {key: {'media':np.mean(image), 'desviacion':np.std(image)}
                 for key, image in self.datos.img_dict.items() if key !='textura'}

        histo_canva = plt.figure(figsize=(10, 7))

        for i, (key, image) in enumerate(self.datos.img_dict.items()):
            # if i<4:
            if key != 'textura':
                ax_img = histo_canva.add_subplot(4, 3, 3 * i + 1)
                #img
                imagen = ax_img.imshow(image, cmap='gray')
                ax_img.set_title(f'imagen {key}')
                ax_img.axis('off')
                # histograma
                ax_hist = histo_canva.add_subplot(4, 3, 3 * i + 2)
                hist = ax_hist.hist(image.ravel(), bins=256, range=[10, 256], color='gray', alpha=0.75)
                ax_hist.set_title(f'histograma imagen {key}')

                # metricas
                ax_stats=histo_canva.add_subplot(4, 3, 3 * i + 3)
                stats_txt=(f'Media:{stats[key]['media']:.2f} \n '
                           f'Desviacion estándar:{stats[key]['desviacion']:.2f}')
                ax_stats.text(0.5,0.5,stats_txt,horizontalalignment = 'center',verticalalignment='center',fontsize = 12 )
                ax_stats.axis('off')


            else:
                break

        # histo_canva.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.3)
        plt.show()


#
# img_rutas = {'top': 'imagenes/SENOS1-T.BMP', 'bottom': 'imagenes/SENOS1-B.BMP', 'left': 'imagenes/SENOS1-L.BMP',
#              'right': 'imagenes/SENOS1-R.BMP', 'textura': 'imagenes/SENOS1-S.BMP'}

img_rutas = {'top': 'imagenes/4-C-T.BMP', 'bottom': 'imagenes/4-C-B.BMP', 'left': 'imagenes/4-C-L.BMP',
             'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}

# img_rutas = {'top': 'imagenes/6-C-T.BMP', 'bottom': 'imagenes/6-C-B.BMP', 'left': 'imagenes/6-C-L.BMP',
#              'right': 'imagenes/6-C-R.BMP', 'textura': 'imagenes/6-C-S.BMP'}
# img_rutas = {'top': 'imagenes/6M-C-T.BMP', 'bottom': 'imagenes/6M-C-B.BMP', 'left': 'imagenes/6M-C-L.BMP',
#               'right': 'imagenes/6M-C-R.BMP', 'textura': 'imagenes/6M-C-S.BMP'}

# img_rutas = {'top': 'imagenes/CIRC1_T.BMP','bottom': 'imagenes/CIRC1_B.BMP','left': 'imagenes/CIRC1_L.BMP','right': 'imagenes/CIRC1_R.BMP','textura': 'imagenes/CIRC1.BMP'}

# img_rutas = {'top': 'imagenes/RUEDA1_T.BMP','bottom': 'imagenes/RUEDA1_B.BMP','left': 'imagenes/RUEDA1_L.BMP','right': 'imagenes/RUEDA1_R.BMP','textura': 'imagenes/RUEDA1_S.BMP'}
# img_rutas = {'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}


# plt.ion()

cargar = Cargarimagenes(img_rutas)
# histograma = Histograma(cargar)
# procesar = Procesarimagenes(cargar)
ecualizar = Ecualizacion(cargar)
# histograma = Histograma(cargar)
# aplanar = Metodosaplanacion(cargar)
procesar = Procesarimagenes(cargar)


reconstruir = Reconstruccion(cargar)
# contornear = Contornos(reconstruir)


