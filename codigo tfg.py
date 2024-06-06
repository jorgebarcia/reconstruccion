import cv2
import matplotlib
import sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid, trapezoid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.linalg import lstsq
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import linalg
import time
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import bicgstab, spsolve
from scipy.sparse.linalg import cg

matplotlib.use('TkAgg')
# import scienceplots
# plt.style.use(['science', 'grid'])
# plt.style.use('science')


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

# plt.style.use('bmh')
# notebook con un toqucito de ggplot a ver q tal
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

class Cargarimagenes:
    def __init__(self, img_rutas):
        # self.img_rutas = img_rutas
        self.img_dict = {}
        self.textura = None
        self.upload_img(img_rutas)

    def upload_img(self, img_rutas):
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
        self.datos = datos

        'atributos nuevos'
        self.ruido = None

        'aplicamos funciones'
        # self.nivel_ruido()
        self.filtro(ver=False)
        # self.aplicar_fourier(ver=False)
        # self.filtro(ver=False)

    def nivel_ruido(self):
        '''
        Funcion para examinar el nivel de ruido de nuestras imagenes,
        en base a el se escoje un nivel de filtro mas adecuando
        input: Nuestro objeto, mas en concreto el diccionario
        output: el ruido de nuestra imagen
        '''
        print("Valores Ruido")
        result_ruido = {}
        for key, image in self.datos.img_dict.items():
            # if key != 'textura':  --> no hace falta, en esta nueva version no hay textura en img_dict(self.textura)
            self.ruido = np.std(image)
            print(f'El nivel de ruido de {key} es {self.ruido}')
            result_ruido[key] = self.ruido

        return self.ruido

    # def filtro(self, sigma = 20, ver = True):
    def filtro(self, sigma=3, ver=True):
        '''
        Funcion que aplica un filtro gaussiano a nuestras imagenes
        Sabemos que q el gaussiano se adapta bien por a los histogramas
        sigma: nivel de agresividad del filtro
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
        t_fourier=fft2(image) # calcualo de la transformada
        t_fourier=fftshift(t_fourier) #movemos las frecuencias bajas al el medio del espectro
        return t_fourier

    def filtro_trans_inversa(self,t_fourier,r):
        row, col = t_fourier.shape
        mid_row,mid_col= row//2 , col//2 # mid_fila,mid_col= int(fila), int(col)

        #hacemos el filtro passo-basso circular
        mask=np.zeros((row,col),np.uint8)
        centro=[mid_row,mid_col]
        x, y = np.ogrid[:row,:col] #mallado
        mask_area=(x-centro[0])**2 + (y-centro[1])**2 <=r**2 #(x-x0)^2+(y-y0)^2=r^2 por que es circular
        mask[mask_area]=1 #aplicamos el filtro y dejamos que pasen las frecuencias de dentro

        t_fourier_mask=t_fourier*mask #aplicamos la mascara, filtramos las de fuera de r
        inv_t_fourier=ifftshift(t_fourier_mask) # se invierte el espectro con la trans inversa
        img_trans=np.abs(ifft2(inv_t_fourier))  #asi nos aseguramos que sea una imagen REAL
        return img_trans
    def aplicar_fourier(self,ver=True):
        '''
        Funcion que aplica la transformada de fourier sobre self.img_dict
        :param ver: ==True --> vemos la imagen transformada // ==False --> no hay plot
        SSIM: cuanto mas cercano a 1 mejor
        PSNR: cuanto mayor mejor, si es superior a 30+- podemos decir que hasido un exito
        '''
        print('\n Fourier: \n --------------\n')
        for key, image in self.datos.img_dict.items():
            t_fourier=self.transformada_fourier(image)
            # r=self.estimacion_radio(t_fourier)
            # r=self.estimacion_radio()
            r=200
            image_trans = self.filtro_trans_inversa(t_fourier,r)

            #metricas de calidad de la imagen:
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
        de varianzas de cada 'key'
        """
        varianzas = {}
        for key, image in self.datos.img_dict.items():
            varianzas[key] = np.var(image)
        return varianzas

    def calcular_varianzas_todas(self):
        '''
        :return: Nos devuelve el valor de la varianza max y minima basandose en el conjunto de todas las imagenes
        '''
        varianzas = self.calcular_varianzas()
        valores_varianza = list(varianzas.values())

        self.varianza_min = min(valores_varianza)
        self.varianza_max = max(valores_varianza)

    def estimacion_radio(self, image):
        # solo si varianzas estan definidas asi evitamos errores que hay muchas funciones que aplicar
        if not hasattr(self, 'varianza_min') or not hasattr(self, 'varianza_max'):
            self.calcular_varianzas_todas()

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
        self.datos = datos

        "funciones"
        self.ecualizar()
    def contraste(self,image):
        '''
        igual que la del ruido... pero para ecualizar
        '''
        return np.std(image)

    def entropia(self,image):
        hist,_=np.histogram(image.flatten(),bins=256,range=(0,256))
        hist_norm=hist/hist.sum() # normalizamos
        #entropia
        S = -np.sum(hist_norm * np.log2(hist_norm + np.finfo(float).eps)) #np.finfo para evitar log 0
        return  S


    def ecualizar(self):
        print("\n Valores ecualizacion : \n ----------- \n")
        for key, image in self.datos.img_dict.items():
            # print(f"imagen {key}")

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
        Esta funcion lo que hace es aplanar cada una de las imagenes, para así corregir la desviación planar.
        aplanarlas, consiste en hacer un ajuste por mínimos cuadrados a cada una de las imagenes para así restarselo
        y corregir posibles desviaciones de los detectores BSE.

        No funciona bien --> el problema de desviacion planar (inclinacion) sucede en la rep 3D no antes

        Explicacion por que no funciona (2D): restamos f(x)=x a una 'imagen' que es y0=1,y1=2,y2=1. La funcion f
        es estrictamente creciente, por lo que va incrementar las diferencias estre y1 e y2 --> no es correcto!
        además si y2- f(2) se convierte en un número negativo, tras la integración obtendremos valores muy poco acertados

        El fallo es ese, por mas que lo intente no conseguí corregirlo aunque la solucion se ve muy facil (suma el valor
        mas bajo al array y ya está, todo >0 )--> sigue sin funcionar. Parece que la rep3D es bastante
        acertada pero la escala esta mal
        '''

        index_y, index_x = np.indices(image.shape)  # obtenemos dos matrices con indices cuya forma=imagen.shape
        image_flat = image.flatten()  # array.dim=1 (valores planos imagen)

        matriz_sist = np.column_stack((index_x.flatten(), index_y.flatten(), np.ones_like(
            image_flat)))  # consume mas meemoria pero es mejor --> np.ones((imagen_flat.shape.. size))

        # z=c1*x+c2*y+c0, c0 es np.ones_like ya que son los valores de intensidad aplanados
        # A es una matriz que cada fila representa un punto x,y + un termino intependiente
        'realizamos el ajuste por minimos cuadrados'

        # mirar el condicionamiento de nuestraas matrices
        coefi, _, _, _ = lstsq(matriz_sist, image_flat,
                               lapack_driver='gelsy')  #solo queremos llstsq[0]--> array.len=3 con coef c1,c2,c0

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
            #Condicionamiento
            plt.figure()
            plt.title("Numero de condicion")
            plt.bar(['Matriz A'], [numero_condicion], color='blue')
            plt.ylabel('Numero')
            plt.yscale('log')
            plt.show()

        return numero_condicion

    def evaluar_ajuste(self, image, coefi, matriz_sist):
        '''
        Funcion que evalua que tan bien se ha ajustado nuestro plano a las imagenes
        '''
        z_pred = matriz_sist @coefi

        # residuales (diferencia entre los valores reales y los predichos)
        residuos = image.ravel() - z_pred

        # Calcula la norma de los residuales
        norma_residual = np.linalg.norm(residuos)
        print(f"La norma del residuo es: {norma_residual}")

        if norma_residual < 1e-1: #mas o menos, si es pequeño bien, si es grande malo
            print("El ajuste es bueno.")
        else:
            print("El ajuste es malo.")

        return norma_residual


class Reconstruccion:
    def __init__(self,datos):
        self.datos = datos

        "Calibración"
        # self.dpixel = 1 / 251 # piezas bj (primera sesion)
        # self.dpixel = 500/251 # piezas FDM (15 abril)
        self.dpixel = 0.9598 #pixels/um #calibracion
        # calibracion (04 --> a'=a*0.853) ! 06 -->

        'Gradientes --> siempre activa'
        self.calculo_gradientes(1,1,eps=1e-5, ver=False) #c=85.36, d=100

        "Tipos de integración de gradientes"
        # self.integracion( z0=0, ver=True)
        # self.integrar_bidireccional( z0=0, ver=True)
        # self.integrar_poisson()
        # self.integrar_poisson_bicgstab()
        # self.integrar_poisson_spsolve()
        # self.integrar_poisson_condiciones()
        # self.integrar_poisson_neumann()

        self.ny, self.nx = self.datos.img_dict['right'].astype(np.float32).shape
        self.integrar_minima_energia()


        "Tipos de corrección (desviación planar)"
        # self.corregir_plano()
        # self.corregir_polinomio()

        "Reconstrucción y visualizador 3D"
        self.plot_superficie(ver_textura=True)

    def calculo_gradientes(self,c,d, eps=1e-5, ver=True):
        '''
        Funcion que calcula los gradientes independientemente del uso posterior que les demos
        :param eps: restricccion de /0
        :param ver: ver los gradientes en mapa de calor
        :return:
        '''
        # print(self.img_dict)

        # extraemos los datos del diccionario con nomenclatura de acuerdo al paper 'palusky2008'
        i_a = self.datos.img_dict['right'].astype(np.float32)
        i_b = self.datos.img_dict['left'].astype(np.float32)
        i_c = self.datos.img_dict['top'].astype(np.float32)
        i_d = self.datos.img_dict['bottom'].astype(np.float32)
        # print(i_a.dtype)
        factor = c/d
        # cálculo de los gradientes con restriccion de 0 (aunque ya nunca sucede /0, en mis primeras pruebas si)
        self.s_dx = factor * (i_a - i_b) / np.clip(i_a + i_b, eps, np.inf)
        self.s_dy = factor * (i_d - i_c) / np.clip(i_c + i_d, eps, np.inf)

        if ver:
            figs=[i_a, i_b,i_a - i_b, i_d, i_c, i_a - i_b]
            titulos=['i_a', 'i_b','i_b - i_a', 'i_d','i_c', 'i_d - i_c']
            fig_grad=plt.figure(figsize=(8,5))
            for i in range(6):
                axs=fig_grad.add_subplot(2,3,i+1) # misma sintaxis que para hist
                plot=axs.imshow(figs[i],cmap='viridis')
                axs.set_title(titulos[i])
                axs.axis('off')
            fig_grad.colorbar(plot,ax=axs,orientation='vertical')
            plt.show()


    def integracion(self,z0):

        # integramos los gradientes segun el metodo acumulativo de los trapecios
        z_x = cumulative_trapezoid(self.s_dx, dx=self.dpixel, axis=1, initial=z0)
        z_y = cumulative_trapezoid(self.s_dy, dx=self.dpixel, axis=0, initial=z0)

        # z_x = trapezoid(s_dx*c/d,dx=self.dpixel,axis=0)
        # z_y = trapezoid(s_dy,dx=self.dpixel,axis=1)

        self.z = z_x + z_y  # ahora self.z ya no es None

        # metricas que no nos dicen mucho a no ser que sepamos que estamos trabajando con una superficie totalmente plana
        media = self.z.mean()
        desviacion = self.z.std()
        print('\n Valores integracion (cumtrapz): \n --------------\n')
        print(f'Valor medio: {media}')
        print(f'Desviacion: {desviacion}')


    def integrar_bidireccional(self,z0):
        '''
        Funcion que integra bidireccionarlmente los gradientes,
        a partir de ensayo y error, por la variabilidad de los detectores:
        'bien pa 04'
        # s_dx = (i_a - i_b) / (i_a + i_b)
        # s_dy = (i_d - i_c) / (i_c + i_d)

        # '04 al reves'
        # s_dx = (i_b - i_a) / (i_a + i_b)
        # s_dy = (i_c - i_d) / (i_c + i_d)

        # 'o6 bien'
        # s_dx = (i_a - i_b) / (i_a + i_b)
        # s_dy = (i_c - i_d) / (i_c + i_d)

        # '06 reves'
        # s_dx = (i_b - i_a) / (i_a + i_b)
        # s_dy = (i_d - i_c) / (i_c + i_d)
        :return: self.z --> alturas de la reconstruccion
        '''

        # Integración de izquierda a derecha en x
        z_lr = np.cumsum(self.s_dx * self.dpixel , axis=1)
        z_lr=np.flip(z_lr,axis=0)
        # Integración de derecha a izquierda en x (invierte la matriz, integra, y luego invierte el resultado)
        z_rl = np.cumsum(np.flip(self.s_dx, axis=1) * self.dpixel , axis=1)
        z_rl = np.flip(z_rl, axis=0)

        # Integración de arriba a abajo en y
        z_tb = np.cumsum(self.s_dy * self.dpixel, axis=0)
        z_tb = np.flip(z_tb, axis=1)
        # Integración de abajo hacia arriba en y (invierte la matriz, integra, y luego invierte el resultado)
        z_bt = np.cumsum(np.flip(self.s_dy, axis=0) * self.dpixel, axis=0)
        z_bt = np.flip(z_bt, axis=1)

        # combinamos en self.z
        self.z = (-z_lr + z_rl - z_tb + z_bt) / 4

        media= self.z.mean()
        desviacion=self.z.std()

        print('\n Valores integracion (bidireccional;cumsum): \n --------------\n')
        print(f'la media es {media}')
        print(f'la desviacion es {desviacion}')

        # return self.z


    def integrar_poisson(self):
        start_time=time.time()

        i_a = self.datos.img_dict['right'].astype(np.float32)
        ny, nx = i_a.shape


        # sistema lineal para la integración de Poisson
        N = ny * nx
        A = lil_matrix((N, N))
        b = np.zeros(N)

        # completamos mat A y b
        for j in range(ny):
            for i in range(nx):
                index = j * nx + i
                b[index] = -((self.s_dx[j, i] if i < nx - 1 else 0) - (self.s_dx[j, i-1] if i > 0 else 0) +
                             (self.s_dy[j, i] if j < ny - 1 else 0) - (self.s_dy[j-1, i] if j > 0 else 0))

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
        self.z=z

        end_time = time.time()
        print(f"Tiempo de ejecución de la integración de poisson: {end_time - start_time} segundos")

        # return self.z

    def integrar_poisson_condiciones(self, ver=True):
        start_time = time.time()

        i_a = self.datos.img_dict['right'].astype(np.float32)
        ny, nx = i_a.shape

        # sistema lineal para la integración de Poisson
        N = ny * nx
        A = lil_matrix((N, N))
        b = np.zeros(N)

        # completamos mat A y b
        for j in range(ny):
            for i in range(nx):
                index = j * nx + i
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    # Aplicar condiciones de Dirichlet en los bordes
                    A[index, index] = 1
                    b[index] = 0
                else:
                    # Cálculo de gradientes interiores
                    b[index] = -((self.s_dx[j, i] if i < nx - 1 else 0) - (self.s_dx[j, i - 1] if i > 0 else 0) +
                                 (self.s_dy[j, i] if j < ny - 1 else 0) - (self.s_dy[j - 1, i] if j > 0 else 0))

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
        self.z = z

        end_time = time.time()
        if ver:
            print(f"Tiempo de ejecución de la integración de poisson: {end_time - start_time} segundos")

        return self.z

    def integrar_poisson_neumann(self, gradiente_borde=0.1):
        start_time = time.time()

        i_a = self.datos.img_dict['right'].astype(np.float32)
        ny, nx = i_a.shape

        N = ny * nx
        A = lil_matrix((N, N))
        b = np.zeros(N)

        for j in range(ny):
            for i in range(nx):
                index = j * nx + i
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    # Aplicar condiciones de Neumann en los bordes
                    A[index, index] = 1
                    A[index, index + (1 if i < nx - 1 else -1)] = -1  # ejemplo simple
                    b[index] = gradiente_borde
                else:
                    # Cálculo de gradientes interiores como antes
                    b[index] = -(self.s_dx[j, i] - self.s_dx[j, i - 1] + self.s_dy[j, i] - self.s_dy[j - 1, i])
                    A[index, index] = 4
                    if i > 0:
                        A[index, index - 1] = -1
                    if i < nx - 1:
                        A[index, index + 1] = -1
                    if j > 0:
                        A[index, index - nx] = -1
                    if j < ny - 1:
                        A[index, index + nx] = -1

        A_csr = A.tocsr()
        z_vector, _ = cg(A_csr, b)
        z = z_vector.reshape(ny, nx)
        self.z = z

        end_time = time.time()

        print(f"Tiempo de ejecución de la integración de poisson: {end_time - start_time} segundos")

        return self.z

    def energia(self, z):
        """ Calcula la energía total incluyendo la fidelidad de datos y el término de regularización. """
        z = z.reshape(self.ny, self.nx)
        gz = np.gradient(z)
        energia_total = np.sum((self.datos - z) ** 2) + self.alpha * np.sum(gz[0] ** 2 + gz[1] ** 2)
        return energia_total

    def gradiente_energia(self, z):
        """ Calcula el gradiente de la función de energía con respecto a z. """
        z = z.reshape(self.ny, self.nx)
        grad = np.zeros_like(z)
        gz = np.gradient(z)
        gzx, gzy = gz
        grad_gz = np.gradient(gzx)[0] + np.gradient(gzy)[1]
        grad += -2 * (self.datos - z) + 2 * self.alpha * grad_gz
        return grad.ravel()

    def integrar_minima_energia(self, alpha=0.1):
        """ Función para iniciar la optimización utilizando un método adecuado para la minimización. """
        self.alpha = alpha  # Coeficiente de regularización
        z_inicial = np.zeros(self.ny * self.nx)  # Valor inicial de z, aplana la matriz inicial

        resultado = minimize(
            fun=self.energia,
            x0=z_inicial,
            method='L-BFGS-B',
            jac=self.gradiente_energia,
            options={'disp': True}
        )

        self.z = resultado.x.reshape(self.ny, self.nx)
        return self.z


    def integrar_poisson_bicgstab(self, eps=1e-5):
        """
        Esta función integra los gradientes de la imagen usando el método de Poisson para la reconstrucción de la superficie.
        Utiliza el método BiCGSTAB para resolver el sistema lineal, que es más adecuado para matrices que no son simétricas
        o que están mal condicionadas.
        """
        start_time = time.time()

        # Asignación de gradientes precalculados
        s_dx = self.s_dx
        s_dy = self.s_dy

        ny, nx = s_dx.shape
        N = ny * nx

        # Crear matrices directamente en formato CSR
        data = []
        row_ind = []
        col_ind = []
        b = np.zeros(N)

        for j in range(ny):
            for i in range(nx):
                index = j * nx + i
                if i > 0:
                    row_ind.append(index)
                    col_ind.append(index - 1)
                    data.append(-1)

                if i < nx - 1:
                    row_ind.append(index)
                    col_ind.append(index + 1)
                    data.append(-1)

                if j > 0:
                    row_ind.append(index)
                    col_ind.append(index - nx)
                    data.append(-1)

                if j < ny - 1:
                    row_ind.append(index)
                    col_ind.append(index + nx)
                    data.append(-1)

                # El elemento diagonal
                row_ind.append(index)
                col_ind.append(index)
                data.append(4)

                # Llenar el vector b con el laplaciano discretizado de los gradientes ajustados por c/d
                b[index] = -((s_dx[j, i] if i < nx - 1 else 0) - (s_dx[j, i - 1] if i > 0 else 0) +
                             (s_dy[j, i] if j < ny - 1 else 0) - (s_dy[j - 1, i] if j > 0 else 0))

        A_csr = csr_matrix((data, (row_ind, col_ind)), shape=(N, N))

        # Resolver el sistema lineal usando el método BiCGSTAB
        z_vector, exit_code = bicgstab(A_csr, b)
        z = z_vector.reshape(ny, nx)

        end_time = time.time()
        print(f"Tiempo de ejecución de la integración de Poisson bicgstab: {end_time - start_time} segundos")
        print(f"Código de salida BiCGSTAB: {exit_code}")

        self.z = z
        return z


    def integrar_poisson_spsolve(self):
        start_time = time.time()

        # Asumimos que los gradientes ya están almacenados en self.s_dx y self.s_dy
        s_dx = self.s_dx
        s_dy = self.s_dy

        ny, nx = s_dx.shape

        # Preparar sistema lineal para la integración de Poisson
        N = ny * nx
        A = lil_matrix((N, N))
        b = np.zeros(N)

        # Rellenar la matriz A y el vector b
        for j in range(ny):
            for i in range(nx):
                index = j * nx + i
                b[index] = -((s_dx[j, i] if i < nx - 1 else 0) - (s_dx[j, i - 1] if i > 0 else 0) +
                             (s_dy[j, i] if j < ny - 1 else 0) - (s_dy[j - 1, i] if j > 0 else 0))

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

        # Resolver sistema lineal usando un método más eficiente
        z_vector = spsolve(A_csr, b)
        z = z_vector.reshape(ny, nx)
        end_time = time.time()
        print(f"Tiempo de ejecución de la integración de Poisson: {end_time - start_time} segundos")
        self.z = z
        # return z


    def corregir_plano(self):
        z = self.z

        #mallado en funcion de z
        x_index, y_index = np.indices(z.shape)

        # aplanamos la malla para la regresión lineal
        X = np.stack((x_index.ravel(), y_index.ravel(), np.ones_like(x_index).ravel()), axis=-1)

        # variable respuesta -->
        Y = z.ravel()

        # ajustamos el plano a los datos
        coeficientes, residuos, rank, s = linalg.lstsq(X, Y)

        # calculamos el plano estimado usando coeficientes
        plano_estimado = X @ coeficientes
        plano_estimado = plano_estimado.reshape(z.shape)

        # restamos el plano estimado de la topografía original para corregir la inclinación --> ojo que dependiendo de los detectores --> gradientes igual hay que sumarlo
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

        # métricas

        print('\n Valores correccion desviacion planar (plano): \n --------------\n')
        print("Coeficientes del plano:", coeficientes)
        print("Suma de residuos cuadrados:", residuos)

        # Calcular métricas adicionales
        mse = np.mean((z - z_corregido) ** 2)
        mae = np.mean(np.abs(z - z_corregido))
        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)

        # return z_corregido, plano_estimado, residuos, coeficientes


    def corregir_polinomio(self, grado=3):
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
        if Rz is not None: print(f'Rz = {Rz}')

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
# img_rutas = {'top': 'imagenes/SENOS1-T.BMP', 'bottom': 'imagenes/SENOS1-B.BMP', 'left': 'imagenes/SENOS1-L.BMP',
#              'right': 'imagenes/SENOS1-R.BMP', 'textura': 'imagenes/SENOS1-S.BMP'}

# img_rutas = {'top': 'imagenes/CIRC1_T.BMP','bottom': 'imagenes/CIRC1_B.BMP','left': 'imagenes/CIRC1_L.BMP','right': 'imagenes/CIRC1_R.BMP','textura': 'imagenes/CIRC1.BMP'}

# img_rutas = {'top': 'imagenes/RUEDA1_T.BMP','bottom': 'imagenes/RUEDA1_B.BMP','left': 'imagenes/RUEDA1_L.BMP','right': 'imagenes/RUEDA1_R.BMP','textura': 'imagenes/RUEDA1_S.BMP'}
# img_rutas = {'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}



# img_rutas = {'top': 'imagenes/4-C-T.BMP', 'bottom': 'imagenes/4-C-B.BMP', 'left': 'imagenes/4-C-L.BMP',
#              'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}
# 'modificando nombres...'
img_rutas = {'top': 'imagenes/4-C-B.BMP', 'bottom': 'imagenes/4-C-T.BMP', 'left': 'imagenes/4-C-L.BMP',
             'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}


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

'se supone que bien, pero hacia arriba...'
# img_rutas = {'top': 'calibrado/0_6-B.BMP', 'bottom': 'calibrado/0_6-T.BMP', 'left': 'calibrado/0_6-L.BMP',
#              'right': 'calibrado/0_6-R.BMP', 'textura': 'calibrado/0_6-S.BMP'}

'hacia abajo'
# img_rutas = {'top': 'calibrado/0_6-T.BMP', 'bottom': 'calibrado/0_6-B.BMP', 'left': 'calibrado/0_6-R.BMP',
#              'right': 'calibrado/0_6-L.BMP', 'textura': 'calibrado/0_6-S.BMP'}


# img_rutas = {'top': 'imagenes/4-C-B.BMP', 'bottom': 'imagenes/4-C-T.BMP', 'left': 'imagenes/4-C-L.BMP',
#              'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}




cargar = Cargarimagenes(img_rutas)
# histograma = Histograma(cargar)
procesar = Procesarimagenes(cargar)
ecualizar = Ecualizacion(cargar)
# histograma = Histograma(cargar)
# aplanar = Metodosaplanacion(cargar)

# procesar = Procesarimagenes(cargar)
#
#
reconstruir = Reconstruccion(cargar)
contornear = Contornos(reconstruir)
perfil=contornear.contornear_y(pos_x=480)
perfi=contornear.contornear_x(pos_y=640)
# perfi=contornear.contornear_x(pos_y=200)
perfil=contornear.contornear_y(pos_x=0)
perfil=contornear.contornear_y(pos_x=10)
perfil=contornear.contornear_y(pos_x=1200)
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