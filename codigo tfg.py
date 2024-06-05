import cv2
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')
from scipy.integrate import cumulative_trapezoid, trapezoid
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
from skimage import io, exposure
# import numpy as np
from scipy.optimize import minimize,check_grad
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import basinhopping
# from mayavi import mlab
import time
# import scienceplots
# plt.rc('text', usetex=True)  # Activar el uso de LaTeX
# plt.rc('font', family='serif')  # Usar fuente tipo serif


# plt.style.use(['science', 'grid'])
# plt.style.use('science')
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamuble=r'\usepackage{cm-super}')
# Si deseas personalizar aún más, puedes ajustar el rcParams
# plt.rcParams.update({
#     'font.size': 12,
#     'axes.titlesize': 14,
#     'axes.labelsize': 12,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'lines.linewidth': 2,
#     'lines.markersize': 6,
#     'figure.figsize': (8, 4),
#     'figure.dpi': 300
# })

#
# plt.style.use('science')
# plt.style.use(['science','notebook'])

# import seaborn as sns  # Seaborn para una paleta de colores más atractiva
# plt.style.use('bmh')
# Aplicar el estilo 'notebook' con un toque de ggplot
# plt.style.use(['seaborn-notebook', 'ggplot'])
# plt.style.use('ggplot')

# plt.rcParams.update({
#     'font.size': 16,      # Tamaño de fuente más grande para mejor lectura
#     'axes.titlesize': 18, # Tamaño de título
#     'axes.labelsize': 18, # Tamaño de etiquetas de ejes
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
#     'legend.fontsize': 14,
#     'lines.linewidth': 3, # Líneas más gruesas
#     'lines.markersize': 10, # Marcadores más grandes
#     'axes.grid': True,    # Habilitar la cuadrícula para mejor orientación
#     'grid.alpha': 0.5,    # Transparencia de la cuadrícula
#     'grid.linestyle': '--', # Estilo de la línea de la cuadrícula
# })
# plt.style.use(['science', 'grid'])
# from scipy.sparse import lil_matrix
# from scipy.sparse import coo_matrix
# from scipy.sparse.linalg import spsolve
#
# from scipy.optimize import minimize
# from scipy.spatial.distance import cdist
# from sklearn.metrics import mean_squared_error, mean_absolute_error
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
        self.ruido = None

        'aplicamos funciones'
        # self.nivel_ruido()
        self.filtro(ver=False)
        self.aplicar_fourier(ver=False)
        # self.filtro(ver=False)

    def nivel_ruido(self):
        '''
        input: Nuestro objeto, mas en concreto el diccionario
        output: el ruido de nuestra imagen
        '''
        print("Valores Ruido")
        result_ruido = {}
        for key, image in self.datos.img_dict.items():
            # if key != 'textura':  --> no hace falta, en esta nueva version no hay textura en img_dict
            self.ruido = np.std(image)
            print(f'El nivel de ruido de {key} es {self.ruido}')
            # return self.ruido #es necesario??'
            result_ruido[key] = self.ruido
        return result_ruido


    def filtro(self, sigma = 20, ver = True):
    # def filtro(self, sigma=3, ver=True):
        '''
        Funcion que aplica un filtro gaussiano a nuestras imagenes
        Sabemos que tiene ruido gaussiano debido a los histogramas
        sigma: nivel de agresividad edl filtro
        '''
        print("\n Filtro gaussiano: \n ----------- \n")
        for key, image in self.datos.img_dict.items():
            img_no_filtrada = self.datos.img_dict[key]
            self.datos.img_dict[key] = gaussian_filter(image, sigma=sigma)
            print(f'La imagen {key} se ha filtrado')

            if ver == True:
                canva = plt.figure(figsize=(8,3))
                original=canva.add_subplot(121)
                original.imshow(img_no_filtrada, cmap ='gray')
                original.set_title(f'Original {key}')
                original.axis('off')

                filtrada = canva.add_subplot(122)
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
        print('\n Fourier: \n --------------\n')
        for key, image in self.datos.img_dict.items():
            t_fourier=self.transformada_fourier(image)
            # r=self.estimacion_radio(t_fourier)
            # r=self.estimacion_radio()
            r=200
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
        print("\n Valores ecualizacion : \n ----------- \n")
        for key, image in self.datos.img_dict.items():
            # print(f"Procesando imagen {key}")

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
            print(f"Imagen {key} - Mejora de Entropía: {entropia_despues - entropia_antes} \n")

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


        val_max=np.max(image)
        val_min=np.min(image)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Imagen Original \n- Max: {val_max}\n Min: {val_min}')
        plt.colorbar()

        val_max = np.max(plano)
        val_min = np.min(plano)
        plt.subplot(1, 3, 2)
        plt.imshow(plano, cmap='gray')
        plt.title(f'Plano -\n Max: {val_max}\nPlano - Min: {val_min}')
        plt.colorbar()

        val_max= np.max(image_correct)
        val_min = np.min(image_correct)
        plt.subplot(1, 3, 3)
        plt.imshow(image_correct, cmap='gray')
        plt.title(f'Imagen Ajustada -\n Max: {val_max}\n Min: {val_min}')
        # plt.text('')
        plt.colorbar()
        plt.show()


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
        # self.dpixel = 500/251
        # self.dpixel = 1 / 251
        self.dpixel = 0.9598 #pixels/um
        # calibracion 04 --> a'=a*0.8536

        "funciones"
        # self.integracion(85.36,100,0)
        # self.integrar_bidireccional(1,1,0)
        self.integrar_poisson_con_gradientes()

        # self.corregir_plano()
        self.corregir_polinomio()
        self.plot_superficie(ver_textura=True)
    def integracion(self, c, d, z0, eps=1e-5):
        # print(self.img_dict)

        i_a = self.datos.img_dict['right'].astype(np.float32)
        i_b = self.datos.img_dict['left'].astype(np.float32)
        i_c = self.datos.img_dict['top'].astype(np.float32)
        i_d = self.datos.img_dict['bottom'].astype(np.float32)
        figura=plt.figure(figsize=(8,5))
        plt.imshow(i_a-i_b, cmap='viridis')
        plt.colorbar(cmap='viridis')
        plt.show()
        # i_a=i_a/255+1
        # i_b=i_b/255+1
        # i_c=i_c/255+1
        # i_d=i_d/255+1

        # print(i_a.dtype)
        # restriingimos la division por cero para uqe no nos salte ningun error
        s_dx = (i_a - i_b) / np.clip(i_a + i_b, eps, np.inf)
        s_dy = (i_d - i_c) / np.clip(i_c + i_d, eps, np.inf)

        # s_dx = (i_a - i_b) / (i_a+i_b)
        # s_dy = (i_d - i_c) / (i_c+i_d)


        z_x = cumulative_trapezoid(s_dx * c / d, dx=self.dpixel, axis=1, initial=z0)
        z_y = cumulative_trapezoid(s_dy * c / d, dx=self.dpixel, axis=0, initial=z0)


        # z_x = trapezoid(s_dx*c/d,dx=self.dpixel,axis=0)
        # z_y = trapezoid(s_dy,dx=self.dpixel,axis=1)

        self.z = z_x + z_y  # ahora self.z ya no es None

        media = self.z.mean()
        desviacion = self.z.std()
        print('\n Valores integracion (cumtrapz): \n --------------\n')
        print(f'Valor medio: {media}')
        print(f'Desviacion: {desviacion}')

    def integrar_bidireccional(self,c,d,z0, eps=1e-5):
        i_a = self.datos.img_dict['right'].astype(np.float32)
        # print(i_a.dtype)
        i_b = self.datos.img_dict['left'].astype(np.float32)
        i_c = self.datos.img_dict['top'].astype(np.float32)
        i_d = self.datos.img_dict['bottom'].astype(np.float32)
        # totalmente innecesario esto:
        # i_a = i_a / 255
        # i_b = i_b / 255
        # i_c = i_c / 255
        # i_d = i_d / 255

        print(i_a.shape)

        figura = plt.figure(figsize=(8, 5))

        figura.add_subplot(231)
        plt.imshow(i_a, cmap='viridis')
        plt.title('i_a ; right')

        figura.add_subplot(232)
        plt.imshow(i_b, cmap='viridis')
        plt.title('i_b ; left')

        figura.add_subplot(233)
        plt.imshow(i_a-i_b, cmap='viridis')
        plt.title('i_b-i_a')

        figura.add_subplot(234)
        plt.imshow(i_d, cmap='viridis')
        plt.title('i_d ; right')

        figura.add_subplot(235)
        plt.imshow(i_c, cmap='viridis')
        plt.title('i_c ; left')

        figura.add_subplot(236)
        plt.imshow(i_a - i_b, cmap='viridis')
        plt.title('i_c-i_d')

        plt.colorbar(cmap='viridis')
        plt.show()

        # restriingimos la division por cero para uqe no nos salte ningun error
        s_dx = (i_a - i_b) / np.clip(i_a + i_b, eps, np.inf)
        s_dy = (i_d - i_c) / np.clip(i_c + i_d, eps, np.inf)
        'bien pa 04'
        # s_dx = (i_a - i_b) / (i_a + i_b)
        # s_dy = (i_d - i_c) / (i_c + i_d)

        '04 al reves'
        # s_dx = (i_b - i_a) / (i_a + i_b)
        # s_dy = (i_c - i_d) / (i_c + i_d)

        'o6 bien'
        # s_dx = (i_a - i_b) / (i_a + i_b)
        # s_dy = (i_c - i_d) / (i_c + i_d)

        'o6 bien reves'
        # s_dx = (i_b - i_a) / (i_a + i_b)
        # s_dy = (i_d - i_c) / (i_c + i_d)



        # Integración de izquierda a derecha en x
        z_lr = np.cumsum(s_dx * self.dpixel, axis=1)
        z_lr=np.flip(z_lr,axis=0)
        # Integración de derecha a izquierda en x (invierte la matriz, integra, y luego invierte el resultado)
        z_rl = np.cumsum(np.flip(s_dx, axis=1) * self.dpixel, axis=1)
        z_rl = np.flip(z_rl, axis=0)

        # Integración de arriba a abajo en y
        z_tb = np.cumsum(s_dy * self.dpixel, axis=0)
        z_tb = np.flip(z_tb, axis=1)
        # Integración de abajo hacia arriba en y (invierte la matriz, integra, y luego invierte el resultado)
        z_bt = np.cumsum(np.flip(s_dy, axis=0) * self.dpixel, axis=0)
        z_bt = np.flip(z_bt, axis=1)

        # Combinación de las integraciones
        z_combined = (-z_lr + z_rl - z_tb + z_bt) / 4
        self.z=z_combined
        media= z_combined.mean()
        desviacion=z_combined.std()

        print('\n Valores integracion (bidireccional;cumsum): \n --------------\n')
        print(f'la media es {media}')
        print(f'la desviacion es {desviacion}')

        fig, axs = plt.subplots(1, 1, figsize=(15, 5))
        cax_0 = axs.imshow(self.z, cmap='plasma')
        fig.colorbar(cax_0, ax=axs, orientation='vertical')
        axs.set_title('Topografía Original')
        axs.axis('off')

        return self.z

    def integrar_poisson_con_gradientes(self, c=1, d=1, eps=1e-5):
        start_time=time.time()
        from scipy.sparse import lil_matrix, csr_matrix
        from scipy.sparse.linalg import cg
        # Extracción de imágenes del diccionario de datos
        i_a = self.datos.img_dict['right'].astype(np.float32)
        i_b = self.datos.img_dict['left'].astype(np.float32)
        i_c = self.datos.img_dict['top'].astype(np.float32)
        i_d = self.datos.img_dict['bottom'].astype(np.float32)

        ny, nx = i_a.shape

        # Cálculo de gradientes
        s_dx = (i_a - i_b) / np.clip(i_a + i_b, eps, np.inf)
        s_dy = (i_d - i_c) / np.clip(i_c + i_d, eps, np.inf)

        # Preparar sistema lineal para la integración de Poisson
        N = ny * nx
        A = lil_matrix((N, N))
        b = np.zeros(N)

        # Rellenar la matriz A y el vector b
        for j in range(ny):
            for i in range(nx):
                index = j * nx + i
                b[index] = -((s_dx[j, i] if i < nx - 1 else 0) - (s_dx[j, i-1] if i > 0 else 0) +
                             (s_dy[j, i] if j < ny - 1 else 0) - (s_dy[j-1, i] if j > 0 else 0))

                if i > 0:
                    A[index, index - 1] = -1
                if i < nx - 1:
                    A[index, index + 1] = -1
                if j > 0:
                    A[index, index - nx] = -1
                if j < ny - 1:
                    A[index, index + nx] = -1
                A[index, index] = 4

        # Convertir a CSR para la solución eficiente del sistema
        A_csr = A.tocsr()

        # Resolver sistema lineal usando el método del Gradiente Conjugado
        z_vector, _ = cg(A_csr, b)
        z = z_vector.reshape(ny, nx)
        end_time = time.time()
        print(f"Tiempo de ejecución: {end_time - start_time} segundos")
        self.z=z
        return z

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
        # z_corregido = z + plano_estimado
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

        print('\n Valores correccion desviacion planar (plano): \n --------------\n')
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
        # z_corregido = z + z_estimado

        # Calcular errores
        mse = np.mean((z - z_estimado) ** 2)
        mae = np.mean(np.abs(z - z_estimado))
        # plot_3d=plt.figure(figsize=(7,7))

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
        print('\n Valores correccion desviacion planar (polinomio): \n --------------\n')
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
        axis_1.plot_surface(x * self.dpixel, y * self.dpixel, self.z, cmap='plasma',shade=True)

        axis_1.set_title('Topografia sin textura')
        # axis_1.set_xlabel('X (mm)')
        # axis_1.set_ylabel('Y (mm)')
        # axis_1.set_zlabel('Z (mm)')

        axis_1.set_xlabel(r'X $(\mu m)$')
        axis_1.set_ylabel(r'Y $(\mu m)$')
        axis_1.set_zlabel(r'Z $(\mu m)$')

        # axis_1.xlim(-self.dpixel, self.dpixel)
        # axis_1.set_zlim(bottom=-40, top=200)
        # axis_1.set_zticks(np.arange(-20, 40, 20))

        axis_1.get_proj = lambda: np.dot(Axes3D.get_proj(axis_1), np.diag([1.0, 1.0, 0.4, 1]))
        axis_1.tick_params(axis='both', which='major', labelsize=7)

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
            # axis_2.set_xlabel('X (mm)')
            # axis_2.set_ylabel('Y (mm)')
            # axis_2.set_zlabel('Z (mm)')

            axis_2.set_xlabel(r'X $(\mu m)$')
            axis_2.set_ylabel(r'Y $(\mu m)$')
            axis_2.set_zlabel(r'Z $(\mu m)$')
            # axis_2.set_zticks(np.arange(-20, 40, 20))

            axis_2.get_proj = lambda: np.dot(Axes3D.get_proj(axis_2), np.diag([1.0, 1.0, 0.4, 1]))

            axis_2.tick_params(axis='both', which='major', labelsize=7)
            # axis_2.secondary_xaxis()
            axis_2.grid(True)
            # axis_2.set_zlim(bottom=-20, top=40)
            # axis_2.set_zlim(-20,200)

            mappable_gray = cm.ScalarMappable(cmap=cm.gray)
            mappable_gray.set_array(self.z)
            plt.colorbar(mappable_gray, ax=axis_2, orientation='vertical', label='Intensidad', shrink=0.5, pad=0.2)

            plt.show()


class Contornos:
    def __init__(self, Reconstruccion):
        self.z = Reconstruccion.z
        self.dpixel = Reconstruccion.dpixel

        # self.contornear_x(300)
        # self.contornear_y(300)
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
        contorno = plt.figure(figsize=(10,4))
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
        contorno = plt.figure(figsize=(10, 4))
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

        # corchete dz
        alto = 0.05
        ancho = ax_x[-1] - alto * 2
        ax.plot([ancho, ancho], [media_perfil, media_perfil + Ra], 'k-', lw=1)
        ax.plot([ancho - alto / 2, ancho + alto / 2], [media_perfil + Ra, media_perfil + Ra], 'k-', lw=1)
        ax.plot([ancho - alto / 2, ancho + alto / 2], [media_perfil, media_perfil], 'k-', lw=1)
        ax.text(ancho + alto, media_perfil + Ra / 2, f'Δz={Ra:.2f}', va='center', ha='left', backgroundcolor='w')
        # ax.legend()
        # plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
        plt.legend(fontsize='xx-small')
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.title(f'Perfil a lo largo de la posición x= {pos_x}')
        plt.xlabel(r'Distancia $(\mu m)$')
        plt.ylabel(r'Altura $(\mu m)$')
        plt.show()

        print('\n Rugosidades: \n ------------- \n')
        print(f'Ra (media) = {Ra}')
        print(f'Rmax = {Rmax}')
        print(f'Rmin = {Rmin}')
        if Rz is None: None
        else: print(f'Rz = {Rz}')

        return perfil,ax_x
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
        # self.histogramear_solo()
    def histogramear(self):
        stats = {key: {'media':np.mean(image), 'desviacion':np.std(image)}
                 for key, image in self.datos.img_dict.items() if key !='textura'}

        histo_canva = plt.figure(figsize=(10, 9))

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
                sns.hist = ax_hist.hist(image.ravel(), bins=256, range=[10, 254], color='#3F5D7D', alpha=0.75)
                # ax_hist.hist(image.ravel(), bins=256, range=[10, 254], color='#3F5D7D', alpha=0.75)
                ax_hist.tick_params(axis='both', which='major', labelsize=7)
                # ax_hist.grid(False)
                ax_hist.set_title(f'Histograma {key}',fontsize=8, fontweight='bold')
                ax_hist.set_xlabel('Intensidad',fontsize=8, fontweight='bold')
                ax_hist.set_ylabel('Frecuencia',fontsize=8, fontweight='bold')
                ax_hist.grid(False)

                # metricas
                ax_stats=histo_canva.add_subplot(4, 3, 3 * i + 3)
                stats_txt=(f'Media:{stats[key]['media']:.2f} \n '
                           f'Desviacion estándar:{stats[key]['desviacion']:.2f}')
                ax_stats.text(0.5,0.5,stats_txt,horizontalalignment = 'center',verticalalignment='center',fontsize = 12 )
                ax_stats.axis('off')


            else:
                break

        histo_canva.tight_layout(pad=3.0)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.6, wspace=0.3)
        plt.show()

    def histogramear_solo(self):
        stats = {key: {'media': np.mean(image), 'desviacion': np.std(image)}
                 for key, image in self.datos.img_dict.items() if key != 'textura'}

        # histo_canva = plt.figure(figsize=(7, 5))

        for i, (key, image) in enumerate(self.datos.img_dict.items()):
            # if i<4:
            if key != 'textura':
                histo_canva = plt.figure(figsize=(7, 4))
                # histograma
                ax_hist = histo_canva.add_subplot(111)
                sns.hist = ax_hist.hist(image.ravel(), bins=256, range=[10, 254], color='#3F5D7D', alpha=0.75)
                ax_hist.tick_params(axis='both', which='major', labelsize=6)
                # ax_hist.grid(False)
                ax_hist.set_title(f'Histograma {key}', fontsize=12)
                ax_hist.set_xlabel('Valores', fontsize=8)
                ax_hist.set_ylabel('Frecuencia', fontsize=8)
                ax_hist.grid(False)
                plt.show()
            else:
                break

#
img_rutas = {'top': 'imagenes/SENOS1-T.BMP', 'bottom': 'imagenes/SENOS1-B.BMP', 'left': 'imagenes/SENOS1-L.BMP',
             'right': 'imagenes/SENOS1-R.BMP', 'textura': 'imagenes/SENOS1-S.BMP'}

# img_rutas = {'top': 'imagenes/CIRC1_T.BMP','bottom': 'imagenes/CIRC1_B.BMP','left': 'imagenes/CIRC1_L.BMP','right': 'imagenes/CIRC1_R.BMP','textura': 'imagenes/CIRC1.BMP'}

# img_rutas = {'top': 'imagenes/RUEDA1_T.BMP','bottom': 'imagenes/RUEDA1_B.BMP','left': 'imagenes/RUEDA1_L.BMP','right': 'imagenes/RUEDA1_R.BMP','textura': 'imagenes/RUEDA1_S.BMP'}
# img_rutas = {'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}



# img_rutas = {'top': 'imagenes/4-C-T.BMP', 'bottom': 'imagenes/4-C-B.BMP', 'left': 'imagenes/4-C-L.BMP',
#              'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}
# 'modificando nombres...'
# img_rutas = {'top': 'imagenes/4-C-B.BMP', 'bottom': 'imagenes/4-C-T.BMP', 'left': 'imagenes/4-C-L.BMP',
#              'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}


# img_rutas = {'top': 'imagenes/6-C-T.BMP', 'bottom': 'imagenes/6-C-B.BMP', 'left': 'imagenes/6-C-L.BMP',
#              'right': 'imagenes/6-C-R.BMP', 'textura': 'imagenes/6-C-S.BMP'}
# img_rutas = {'top': 'imagenes/6M-C-T.BMP', 'bottom': 'imagenes/6M-C-B.BMP', 'left': 'imagenes/6M-C-L.BMP',
#               'right': 'imagenes/6M-C-R.BMP', 'textura': 'imagenes/6M-C-S.BMP'}

# img_rutas = {'top': 'imagenes/CIRC1_T.BMP','bottom': 'imagenes/CIRC1_B.BMP','left': 'imagenes/CIRC1_L.BMP','right': 'imagenes/CIRC1_R.BMP','textura': 'imagenes/CIRC1.BMP'}

# img_rutas = {'top': 'imagenes/RUEDA1_T.BMP','bottom': 'imagenes/RUEDA1_B.BMP','left': 'imagenes/RUEDA1_L.BMP','right': 'imagenes/RUEDA1_R.BMP','textura': 'imagenes/RUEDA1_S.BMP'}
# img_rutas = {'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}

#
# img_rutas = {'top': 'calibrado/0_4-T.BMP', 'bottom': 'calibrado/0_4-B.BMP', 'left': 'calibrado/0_4-L.BMP',
#              'right': 'calibrado/0_4-R.BMP', 'textura': 'calibrado/0_4-S.BMP'} #son 950 x 1280 pixeles


# img_rutas = {'top': 'calibrado/0_6-T.BMP', 'bottom': 'calibrado/0_6-B.BMP', 'left': 'calibrado/0_6-L.BMP',
#              'right': 'calibrado/0_6-R.BMP', 'textura': 'calibrado/0_6-S.BMP'}
# img_rutas = {'top': 'calibrado/0_6-B.BMP', 'bottom': 'calibrado/0_6-T.BMP', 'left': 'calibrado/0_6-L.BMP',
#              'right': 'calibrado/0_6-R.BMP', 'textura': 'calibrado/0_6-S.BMP'}

# plt.ion()

cargar = Cargarimagenes(img_rutas)
# histograma = Histograma(cargar)
# procesar = Procesarimagenes(cargar)
ecualizar = Ecualizacion(cargar)
# histograma = Histograma(cargar)
# aplanar = Metodosaplanacion(cargar)

procesar = Procesarimagenes(cargar)
#
#
reconstruir = Reconstruccion(cargar)
contornear = Contornos(reconstruir)
perfil=contornear.contornear_y(pos_x=480)
# perfi=contornear.contornear_x(pos_y=640)
perfi=contornear.contornear_x(pos_y=200)
perfi=contornear.pilacontornos_x(20)


'''
cargar = Cargarimagenes(img_rutas)
ecualizar = Ecualizacion(cargar)
procesar = Procesarimagenes(cargar)
reconstruir = Reconstruccion(cargar)
contornear = Contornos(reconstruir)
perfil=contornear.contornear_y(pos_x=300)
'''
'para comparrar los perfiles'

# img_rutas = {'top': 'imagenes/6-C-T.BMP', 'bottom': 'imagenes/6-C-B.BMP', 'left': 'imagenes/6-C-L.BMP',
#              'right': 'imagenes/6-C-R.BMP', 'textura': 'imagenes/6-C-S.BMP'}
# # #
# carga_06=Cargarimagenes(img_rutas)
# ecualizar_06= Ecualizacion(carga_06)
# procesar_06= Procesarimagenes(carga_06)
# reconstruir_06= Reconstruccion(carga_06)
# contornear_06= Contornos(reconstruir_06)
# perfil_06,ax_x= contornear_06.contornear_y(300)
# #
# img_rutas = {'top': 'imagenes/6M-C-T.BMP', 'bottom': 'imagenes/6M-C-B.BMP', 'left': 'imagenes/6M-C-L.BMP',
#               'right': 'imagenes/6M-C-R.BMP', 'textura': 'imagenes/6M-C-S.BMP'}
# #
# carga_06M=Cargarimagenes(img_rutas)
# ecualizar_06M= Ecualizacion(carga_06M)
# procesar_06M= Procesarimagenes(carga_06M)
# reconstruir_06M= Reconstruccion(carga_06M)
# contornear_06M= Contornos(reconstruir_06M)
# perfil_06M,_= contornear_06M.contornear_y(300)
#
#
# plt.figure(figsize=(10, 4))
# plt.plot(ax_x, perfil_06, label='Pieza sin pulido')
# plt.plot(ax_x, perfil_06M, 'r', label='Pieza con Pulido')
# plt.legend()
# plt.title('Comparación de perfiles')
# plt.xlabel(r'Distancia $(\mu m)$')
# plt.ylabel(r'Altura $(\mu m)$')
# plt.tick_params(axis='both', which='major', labelsize=8)
# plt.grid(False)
# plt.show()