import cv2
import matplotlib
import sns
from numpy.linalg import cond
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid, trapezoid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.linalg import lstsq, solve_continuous_lyapunov, solve_discrete_lyapunov
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import linalg
import time
from scipy.ndimage import maximum_filter, minimum_filter
import timeit
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import bicgstab, spsolve
from scipy.sparse.linalg import cg
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
# matplotlib.use('TkAgg')
from scipy.linalg import solve_sylvester
# import scienceplots
# plt.style.use(['science', 'grid'])
# plt.style.use('science')
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

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

start_total=time.time()
class Cargarimagenes:
    def __init__(self, img_rutas):
        # self.img_rutas = img_rutas
        self.img_dict = {}
        self.textura = None
        # self.upload_img(img_rutas)
        # self.add_gaussian_noise()

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

    def add_gaussian_noise(self, mean, sigma):
        """
        Agrega ruido gaussiano a todas las imágenes en img_dict.
        """
        for key, image in self.img_dict.items():
            self.img_dict[key] = self.apply_gaussian_noise(image, mean, sigma)

    @staticmethod
    def apply_gaussian_noise(image, mean, sigma):
        """
        Aplica ruido gaussiano a una imagen.
        """
        gauss = np.random.normal(mean, sigma, image.shape).reshape(image.shape)
        noisy_image = image + gauss
        noisy_image = np.clip(noisy_image, 0, 255)  # Asegurarse de que los valores estén en el rango de 8 bits
        return noisy_image.astype(np.uint8)


class Procesarimagenes:
    def __init__(self, datos):
        self.datos = datos

        'atributos nuevos'
        self.ruido = None

        'aplicamos funciones'
        # self.nivel_ruido()
        # self.filtro(ver=False)
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

    def filtro(self, sigma = 100, ver = True):
    # def filtro(self, sigma=5, ver=True):
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

            if ver:
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
            r=500
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


class Reconstruccion:
    def __init__(self,datos):
        self.datos = datos

        "Calibración"
        self.dpixel = 1 / 251 # piezas bj (primera sesion)
        # self.dpixel = 500/251 # piezas FDM (15 abril)
        # self.dpixel = 1/0.9598 #pixels/um #calibracion
        # self.dpixel=1.9853  #um/pixel ---> FFF
        # self.dpixel=0.7940  #um/pixel ---> BJ
        # calibracion (04 --> a'=a*0.853) ! 06 -->

        'Gradientes --> siempre activa'
        # self.calculo_gradientes(1,1,eps=1e-5, ver=False) #c=85.36, d=100


        'metodos de integracion'
        # self.least_squares()
        # self.tiempos()


        "Reconstrucción y visualizador 3D"
        # self.plot_superficie(ver_textura=True)


    '----------------------GRADIENTES----------------------'
    def calculo_gradientes(self,c,d, eps=1e-5, ver=True):
        '''
        Funcion que calcula los gradientes independientemente del uso posterior que les demos
        :param eps: restricccion de /0
        :param ver: ver los gradientes en mapa de calor
        :return: self.S_dx y self.S_dy --> gradientes que se integrarán
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
        self.s_dy = factor * (i_c- i_d) / np.clip(i_c + i_d, eps, np.inf)

        #Por si queremos ver mapas de calor de los gradientes y asi verificar que la
        # orientacion del plot sea la adecuada y las imagenes estan bien referenciadas
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
        return self.s_dx, self.s_dy


    '----------------------FUNCIONES DE APOYO--------------'
    def orto(self,A, name="A", tol=1e-8):
        '''
        Funcion que verifica la ortogonalidad de una matriz
        :param A: matriz
        :param name:  nombre matriz
        :param tol: para ajustar valores bajos al 0
        :return: True --> ortogonal, False --> no ortogonal
        '''
        n = A.shape[0]
        I = np.eye(n)
        if np.allclose(A.T @ A, I, atol=tol):
            print(f"Orto {name}: True ; {name} es ortogonal.")
        else:
            print(f"Orto {name}: False ; {name} no es ortogonal.")


    def cond(self,A, name="A"):
        '''
        Funcion que calcula el condicionamiento de una matriz
        :param A: matriz
        :param name: nombre matriz
        :return: Condicionamiento mediante un print
        '''
        U, S, Vt = np.linalg.svd(A)
        if S.min() == 0:
            print(f"Numero condicion {name}: infitito")
        else:
            condi = S.max() / S.min()
            print(f"Numero condicion {name}: {condi}.")


    def todo_mat(self,A,name="A",ver=True):
        '''
        Funcion de apoyo cuando quiero verificar ortogonalidad, calcular y ver el condicionamiento
        muy util durante la creacion de nuevos metodos de integración.
        :param A: matriz
        :param name: nombre de la matriz
        :param ver: True --> plot del condicionamiento; False --> no hay plot
        :return: Valores Condicionamiento y verifica la ortogonalidad
        '''
        self.cond(A,name)
        self.orto(A,name)
        if ver:
            self.plot_cond(A, name)


    def plot_cond(self,A, name="A"):
        '''
        Funcion que representa el condicionamiento de matrices en funcion de sus
        valores sigulares
        :param A:  matriz
        :param name: nombre matriz
        :return: plot
        '''

        singu = np.linalg.svd(A, compute_uv=False)
        plt.figure(figsize=(5, 4))
        plt.plot(singu, marker='o', linestyle='-')
        plt.yscale('log')
        plt.title({name})
        plt.xlabel('Valor')
        plt.ylabel('Valores singulares (logscale)')
        plt.grid(True)
        plt.show()


    def calc_P(self,n):
        '''
        Función de apoyo para el metodo householder
        :param n:
        :return: P matriz ortogonal de apoyo
        '''
        v = np.ones((n, 1))
        v[0] = 1 + np.sqrt(n)
        I = np.eye(n)

        vvt = v @ v.T
        # print('')
        # print(vvt)
        vtv = v.T @ v
        # print(vtv)
        P = I - 2 * (vvt / vtv)
        return P

    '-------------OPERADORES DIFERENCIACION------------------'



    def ope_dif2(self,n, h=1):
        D = np.zeros((n, n))
        for i in range(0, n):
            D[i - 1, i] = 1
            D[i, i - 1] = -1

        D[-1, 0] = 0
        D[0, -1] = 0

        D[0, 1] = 4
        D[-1, -2] = -4

        D[0, 2] = -1
        D[-1, -3] = 1

        D[0, 0] = -3
        D[-1, -1] = 3

        D = D / 2
        return D


    def segundadif(self,n):
        # Crear una matriz de ceros de tamaño n x n
        L = np.zeros((n, n))

        # Llenar la matriz con los valores adecuados para la segunda diferencia
        for i in range(n):
            if i > 0:
                L[i, i - 1] = 1
            L[i, i] = -2
            if i < n - 1:
                L[i, i + 1] = 1
        return L


    def diffmat2(self,n, xspan):
        '''
        Funcion utilizada para la regularización de tikhonov para la creacion de
        un operador diferencial de 1er orden y un ode segundo orden, función extraidda de la bibliografía
        y convertida a lenguaje de python (estaba en matlab)
        :param n: tamaño-1 de la matriz
        :param xspan: rango de adaptacion de los datos
        :return: Dx y Dxx operadores diferenciales
        '''
        a, b = xspan
        h = 1
        x = np.linspace(a, b, n + 1)  # nodes

        # Define most of Dx by its diagonals
        dp = np.full(n, 0.5 / h)  # superdiagonal
        dm = np.full(n, -0.5 / h)  # subdiagonal
        D_x = np.diag(dm, -1) + np.diag(dp, 1)

        # Fix first and last rows
        D_x[0, :3] = np.array([-1.5, 2, -0.5]) / h
        D_x[-1, -3:] = np.array([0.5, -2, 1.5]) / h

        # Define most of D_xx by its diagonals
        d0 = np.full(n + 1, -2 / h ** 2)  # main diagonal
        dp = np.full(n, 1 / h ** 2)  # super- and subdiagonal
        D_xx = np.diag(dp, -1) + np.diag(d0, 0) + np.diag(dp, 1)

        # Fix first and last rows
        D_xx[0, :4] = np.array([2, -5, 4, -1]) / h ** 2
        D_xx[-1, -4:] = np.array([-1, 4, -5, 2]) / h ** 2

        return D_x, D_xx

    '-----------------METODOS DE INTEGRACION-----------------'
    def householder(self,sdx,sdy,Lx,Ly):
        '''

        :param Lx:
        :param Ly:
        :return:
        '''
        print('\n Integracion mediante el método HOUSEHOLDER... \n --------------------------- \n')
        start_house=time.time()
        # m, n = self.s_dx.shape

        n=Lx.shape[0] #ambas son nxn o mxm
        m=Ly.shape[0]
        # self.todo_mat(Lx, 'Lx',False)
        # self.todo_mat(Ly, 'Ly',False)

        #Matrices ortogonales, sirven de apoyo
        Px= self.calc_P(n)
        Py = self.calc_P(m)
        # self.orto(Px,'Px')
        # self.orto(Py,'Py')

        #caluamos L gorro --> primer elemento nulo y condicionamiento menor
        Lxg = Lx @ Px
        Lyg = Ly @ Py
        # self.todo_mat(Lxg,'Lxg',False)
        # self.todo_mat(Lyg,'Lyg',False)

        #imponemos que sean 0 los lugares que son aproximadamente (mejora muy levemente el codnicionamiento)
        tol = 1e-5
        Lxg[np.abs(Lxg) < tol] = 0
        Lyg[np.abs(Lyg) < tol] = 0

        #Calulamos los terminos del sistema de lyapuvnov
        lytly = Lyg.T @ Lyg
        lxtlx = Lxg.T @ Lxg

        #termino independiente (Q)
        G1 = Lyg.T @ sdy @ Px
        G2 = Py.T @ sdx @ Lxg
        G = -(G1 + G2)

        #Extraemos las submatrices para resolver el sistema lineal (bibliografia)
        A = lytly[1:, 1:]
        B = lxtlx[1:, 1:]
        c10 = G[1:, 0]
        c01 = G[0, 1:]
        C = G[1:, 1:]

        # self.cond(A,'A')
        # self.cond(B,'B')

        # A y B int¡vertibles, el sistema es lineal

        # print('sistema w01:')
        w01 = np.linalg.solve(B, -c01.T)
        # print(np.allclose(np.dot(w01.T, B) + c01, 0, atol=1e-10))  # Deberia retornar True si esta correcto

        # print('sistema w10:')
        w10 = np.linalg.solve(A, -c10)
        # print(np.allclose(np.dot(w10, A) + c10, 0, atol=1e-10))

        #Se usó en linalg.lstsq y .solve y dio mejores resultados .solve a pesar del cond A y B
        # error_w01 = w01.T @ B + c01.T
        # error_w10 = A @ w10 + c10

        #resolvemos el termino W11 segun la nueva formulacion Sylvester:
        W11 = solve_sylvester(A, B, -C)
        error = A @ W11 + W11 @ B + C
        # print("Error de la solución Householder:", error)  #muy bajo, es 0

        #reconstruimos W... W[0,0]=0 sería como Z0, nos d aigual
        W = np.zeros((len(w10) + 1, len(w01) + 1))
        W[0, 1:] = w01
        W[1:, 0] = w10
        W[1:, 1:] = W11

        # Px_inv = np.linalg.inv(Px.T)
        # Py_inv_T = np.linalg.inv(Py)

        #Resolvemos Z
        Z = Py @ W @ Px.T
        self.z=Z
        end_house=time.time()
        # print(f'\n Se ha aplicado el metodo householder con una duracion de:{end_house-start_house}')
        # return Z, lytly,lxtlx,G,error,end_house-start_house
        return Z, error, end_house - start_house
        # return Z

    def solve_sylvester(self,sdx,sdy,Lx,Ly):
        '''

        :param Lx:
        :param Ly:
        :return:
        '''
        print('Aplicando integracion mediante sylvester sin regulaciones....')
        start_syl=time.time()
        A = Ly.T @ Ly
        B = Lx.T @ Lx
        C = - (Ly.T @ sdy + sdx @ Lx)

        Z = solve_sylvester(A, B, -C)
        error = A @ Z + Z @ B + C
        # print("Error de la solución Lyapunov:", error)

        self.z = Z
        end_syl = time.time()
        # print(f'Duracion integracion por sylvester: {end_syl-start_syl}')
        return Z,error, end_syl-start_syl
        # return Z

    def reg_dirichlet(self,sdx,sdy,Lx,Ly):
        print('Aplicando regularizacion de dirichlet...')
        m, n = self.s_dx.shape
        P = np.zeros((m, m))
        if m > 2:
            P[1:m - 1, 1:m - 1] = np.eye(m - 2)

        Q = np.zeros((n, n))
        if n > 2:
            Q[1:n - 1, 1:n - 1] = np.eye(n - 2)

        A = P.T @ Ly.T @ Ly @ P
        B = Q.T @ Lx.T @ Lx @ Q
        C = - P.T @ (Ly.T @ sdy + sdx @ Lx) @ Q

        # print(A)
        # print(B)
        # print(C)

        Z = solve_sylvester(A, B, -C)
        error = A @ Z + Z @ B + C
        # print("Error de la solución Lyapunov:", error)
        self.z = Z
        return Z,error


    def reg_tikhonov(self,sdx,sdy,Lx,Ly,Sx,Sy,lamb=1e-1):
        '''
        Esta funcion calcula una nueva superficie Z a partir de la primera estimada mediante
        la regularizacion de tikhonov
        :param Lx:
        :param Ly:
        :param Z0:
        :param lamb:
        :return:
        '''

        print('Regularizacion de tikhonov....')
        print(f'Hemos usado una lambda:{lamb}')

        # Sy= Ly @ Ly
        # Sx= Lx @ Lx
        # print(Lx.shape, Ly.shape, Sx.shape, Sy.shape)

        # Z0=np.ones((Ly.shape[1],Lx.shape[0]))
        Z0 = np.zeros((Ly.shape[1], Lx.shape[0]))

        'comproaciones'
        # print('Ly\n',Ly)
        # print('Lx\n',Lx)
        # print('Ly.T\n', Ly.T)
        # print('Lx.T\n', Lx.T)
        # print('------------')
        # unoy=Ly.T @ Ly
        # unox=Lx.T @ Lx
        # print('Ly.T @ Ly\n', unoy)
        # print('Lx.T @ Lx\n', unox)
        # print('------------')
        # print('Sy\n', Sy)
        # print('Sx\n', Sx)
        # print('------------')
        # dosy=Sy.T @ Sy
        # dosx=Sx.T @ Sx
        # print('Sy.T @ Sy\n', dosy)
        # print('Sx.T @ Sx\n',dosx)
        # print('------------')
        #
        # tresy = (Sy.T @ Sy)*lamb
        # tresx = (Sx.T @ Sx)*lamb
        # print('Sy.T @ Sy\n', tresy)
        # print('Sx.T @ Sx\n', tresx)
        #
        #
        # cuatroy=unoy+tresy
        # cuatrox=unox+tresx
        # print('cuatroy\n',cuatroy)
        # print('cuatrox\n',cuatrox)
        # print('------------')
        # print('ly', Ly.shape)
        # print('s_dy', self.s_dy.shape)
        # print('sdx', self.s_dx.shape)
        # print('Lx', Lx.shape)


        # tol = 1e-10
        # A[np.abs(A) < tol] =0
        # C[np.abs(C) < tol] = 0
        # B[np.abs(B) < tol] = 0

        # print('A',A)
        # print('B',B)
        # print('C',C)
        A = Ly.T @ Ly + lamb * Sy.T @ Sy
        B = Lx.T @ Lx + lamb * Sx.T @ Sx
        C = -((Ly.T @ sdy + sdx @ Lx) + lamb * (Sy.T @ Sy @ Z0 + Z0 @ Sx.T @ Sx))
        # C = -((Ly.T @ self.s_dy + self.s_dx @ Lx) + lamb * (Sy.T @ Sy @ Z_sint + Z_sint @ Sx.T @ Sx))
        # Zx = convolve2d(Z, Sx, mode='same', boundary='symm')
        # Zy = convolve2d(Z, Sy, mode='same', boundary='symm')
        # Z_sint=solve_sylvester(unoy,unox,-C)
        # error = unoy @ Z_sint + Z_sint @ unox + C
        # error = A @ Z_sint + Z_sint @ B + C
        # print("Error de la solución Lyapunov sinikhonov:\n", error)

        Z = solve_sylvester(A, B, -C)


        error = A @ Z + Z @ B + C
        # print("Error de la solución Lyapunov mediante tikhonov:\n", error)
        self.z=Z
        # self.funcion_coste_tik(self.s_dx,self.s_dy,Lx,Ly,Z0,Z,lamb)
        # return Z
        return Z,error
    '-----------------------INTEGRACION POR MINIMOS CUADRADOS-----------------'
    def least_squares(self):
        m, n = self.s_dx.shape
        '----------------------------SYLVESTER------------------------------'
        # Lx_s = self.ope_dif3(n)
        # Ly_s = self.ope_dif3(m)

        # Lx_s = self.ope_dif2(n)
        # Ly_s = self.ope_dif2(m)
        # Z_s= self.solve_sylvester(Lx_s,Ly_s)
        # self.funcion_coste(self.s_dx,self.s_dy,Lx_s,Ly_s,Z_s)
        # self.funcion_coste_tik(self.s_dx, self.s_dy, Lx_s, Ly_s, Z_s,Z_t,1e-1)

        "Solucion con el método -----------HOUSEHOLDER------- mejora la ortogonalidad y el condicionanmiento"
        #de las tres matrices, es la unica que aumenta la funcion coste
        # Lx1 = self.ope_difforward(n)
        # Ly1 = self.ope_difforward(m)
        #
        # Z_house_1 = self.householder(Lx1, Ly1)
        # coste_hou_1 = self.funcion_coste(self.s_dx, self.s_dy, Lx1, Ly1, Z_house_1)

        'La que suelo usar no...?'
        # Lx2=self.ope_dif2(n)
        # Ly2 = self.ope_dif2(m)
        # Z_h = self.householder(Lx2, Ly2)


        # Z_th=self.reg_tikhonov2(Lx2,Ly2,Z_h,1e-2)
        # coste_hou_2 = self.funcion_coste(self.s_dx, self.s_dy, Lx2, Ly2, Z_h)
        # self.funcion_coste_tik(self.s_dx, self.s_dy, Lx2, Ly2, Z_house_2,Z_th,1e-1)

        # Lx3 = self.ope_dif3(n)
        # Ly3 = self.ope_dif3(m)
        # Z_house_3=self.householder(Lx3,Ly3)
        # coste_hou=self.funcion_coste(self.s_dx,self.s_dy,Lx3,Ly3,Z_house_3)



        '-------------------TIKHONOV----------------------'
        # Lx = self.ope_dif2(n)
        # Ly = self.ope_dif2(m)
        # Sx=Lx@Lx
        # Sy=Ly@Ly
        # Lx = self.ope_dif3(n)
        # Ly = self.ope_dif3(m)

        # Lx,Sx = self.diffmat2(n-1,(0,n-1))
        # Ly,Sy = self.diffmat2(m-1,(0,m-1))
        # Lx, Sx = self.diffcheb(n - 1, (0, n - 1))
        # Ly, Sy = self.diffcheb(m - 1, (0, m - 1))

        # Lx, _ = self.diffmat2(n - 1, (0, n - 1))
        # Ly,_ = self.diffmat2(m-1,(0,m-1))
        #
        # Sx=self.segundadif(n)
        # Sy=self.segundadif(m)
        # Z_t=self.reg_tikhonov(Lx,Ly,Sx,Sy,lamb=1e-1)

        # self.reg_tikhonov2(Lx,Ly,lamb=1e-1)
        # self.reg_tikhonov3(Lx, Ly, Sx, Sy, lamb=1e-1)
        # self.corregir_polinomio()
        # self.corregir_plano()



        # Uso de la función
        # optimal_lambda = self.compute_lambda_using_svd(Lx, Ly,Sx,Sy)
        # print("Lambda óptimo calculado:", optimal_lambda)
        # Z_t = self.reg_tikhonov(Lx, Ly, Sx, Sy, optimal_lambda)

        # Z_t=self.reg_tikhonov(Lx,Ly,Sx,Sy,lamb=2e-1)
        # Z_t = self.reg_tikhonov2(Lx,Ly,lamb=1e-3)

        '------------------DIRITCHLET------------------'

        # Lx_d = self.ope_dif2(n)
        # Ly_d = self.ope_dif2(m)

        # Lx_s, Sx = self.diffcheb(n - 1, (0, n - 1))
        # Ly_s, Sy = self.diffcheb(m - 1, (0, m -

        # Z_d = self.reg_dirichlet(Lx_d, Ly_d)


        '-------------MAPA DE CALOR------------'
        # self.mapa_calor(Z_t)

        # z_list = [Z_pol, Z_pla]
        # for i in range(len(z_list)):
        #     self.mapa_calor(z_list[i])
        #     self.plot_superficie(z_list[i], False)

        # z_list = [ Z_t]
        # z_name = [ 'Tikhonov']

        # z_list = [Z_s, Z_t]
        # z_name = ['No regularizada','Tikhonov']

        Lx_s = self.ope_dif2(n)
        Ly_s = self.ope_dif2(m)
        Z_s = self.solve_sylvester(Lx_s, Ly_s)
        #
        Lx2 = self.ope_dif2(n)
        Ly2 = self.ope_dif2(m)
        Z_h = self.householder(Lx2, Ly2)

        Lx, _ = self.diffmat2(n - 1, (0, n - 1))
        Ly, _ = self.diffmat2(m - 1, (0, m - 1))

        Sx = self.segundadif(n)
        Sy = self.segundadif(m)
        Z_t,_,_,_ = self.reg_tikhonov(Lx, Ly, Sx, Sy, lamb=1e-1)

        # self.mapa_calor(Z_t)
        # self.plot_superficie(Z_t)
        #
        Lx_d = self.ope_dif2(n)
        Ly_d = self.ope_dif2(m)
        Z_d = self.reg_dirichlet(Lx_d, Ly_d)
        # self.funcion_coste(self.s_dx,self.s_dy, Lx_d,Ly_d,Z_d)
        # self.plot_superficie(Z_d,False)
        #
        Z_pol=self.corregir_polinomio(Z_s)
        Z_pla=self.corregir_plano(Z_s)
        #
        # z_list = [Z_s, Z_h]
        # z_name=['No regularizada','Householder']

        # z_list = [Z_d, Z_t]
        # z_name = ['Dirichlet', 'Tikhonov']
        #
        # for i in range(len(z_list)):
        #     self.mapa_calor(z_list[i])
        #     self.plot_superficie(z_list[i],False)

        # z_list = [Z_s, Z_h, Z_d, Z_t, Z_pol, Z_pla]
        # z_name = ['No regularizada', 'Householder', 'Dirichlet', 'Tikhonov', 'Polinomio', 'Plano']
        # self.plot_rpsd(z_list,z_name)
        # self.z=Z_s


    '--------------------Analisis de Fourier-------------------'

    def calc_rpsd(self,z,dx):
        #calculamos el psd:
        Z = np.fft.fftshift(np.fft.fft2(z))
        # psd = np.abs(Z) ** 2 / (z.size)  # normalizacion por numero de puntos
        psd = np.abs(Z) ** 2
        psd /= np.sum(psd) #normalizacion por energia total

        #calculamos las k ahora q estamos en su dominio
        kx = np.fft.fftshift(np.fft.fftfreq(z.shape[0], dx))
        ky = np.fft.fftshift(np.fft.fftfreq(z.shape[1], dx))
        # kx = z.shape[0]
        # ky = z.shape[1]
        kx, ky = np.meshgrid(kx, ky)
        kr = np.sqrt(kx ** 2 + ky ** 2)
        # print(kr)
        kr = kr.ravel()
        psd = psd.ravel()

        print(-np.max(kr))
        print(np.max(kr))

        #calculamos el RPSD
        k_bins = np.linspace(0, np.max(kr), num=20)
        k_val = 0.5 * (k_bins[1:] + k_bins[:-1])

        rpsd, _ = np.histogram(kr, bins=k_bins, weights=psd)
        rpsd_norm = rpsd / np.histogram(kr, bins=k_bins)[0]

        # plt.hist(kr, bins=50, log=True)
        # plt.title("Distribución de valores de kr")
        # plt.xlabel("kr (frecuencia radial)")
        # plt.ylabel("Cantidad")
        # plt.show()

        return k_val, rpsd_norm

    def plot_rpsd(self,z_list,name_list):
        plt.figure(figsize=(11, 6))

        for i in range (len(z_list)):
            k_val, RPSD = self.calc_rpsd(z_list[i], 1.042) #1.042
            plt.plot(k_val, RPSD, label=name_list[i])

        plt.axvspan( 1e-2, 4e-2, color='green', alpha=0.1, label='Frecuencias bajas')
        plt.axvspan(4e-2, 1.1e-1, color='blue', alpha=0.1, label='Frecuencias medias')
        plt.axvspan(1.1e-1, 1e1, color='red', alpha=0.1, label='Frecuencias altas')

        # plt.axvline(x=4e-2, color='blue', linestyle='--')
        # plt.axvline(x=1e-1, color='green', linestyle='--')

        plt.axvline(x=4e-2, color='blue', linestyle='--')
        plt.axvline(x=1.1e-1, color='red', linestyle='--')

        # plt.plot(k_val, RPSD, label='Superficie 1')
        # Añadir más según necesario
        plt.xlabel(r'k $(\mu m^{-1})$', fontsize=14)
        plt.ylabel(r'$\hat{W} (m^{3})$', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tick_params(axis='both', which='minor', labelsize=10)

        plt.xscale('log')
        plt.yscale('log')
        # plt.xlim([1e-1, 1e1])
        # plt.xlim([8e-2, 1.2e1])
        # plt.xlim([np.min(k_val) * 0.9, np.max(k_val) * 1.1])
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title('Comparación de RPSDF')
        plt.grid(True)
        plt.show()


    '------------------------FUNCIONES COSTE-------------------'
    def funcion_coste(self,s_dx,s_dy,Dx,Dy,Z):
        '''
        segun Harker, siempre que se conozca la matriz gradiente (S_dx,S_dy), la superficie
        cuyo grandiente es mas cercano en el sentido de min cuadrados globales,
        es la que minimiza la función de costo. formulacion FROBENIUS (F)
        :return: equivalente al error cuadrático medio
        '''
        if self.z is None:
            print("No se ha integrado como para calcular la funcion costo")
            return None
        else:
            c_1=np.linalg.norm(Z @ Dx.T - s_dx,'fro')**2
            c_2=np.linalg.norm(Dy @ Z -s_dy,'fro')**2
            coste = c_1 + c_2
            print(Z.shape)
            print(f'El valor de la funcion coste ha sido de:\n'
                  f'Z*Lx^T - Z_x: {c_1} \n'
                  f'Z*Ly^T - Z_y: {c_2}\n'
                  f'coste: {coste} \n')
        return coste

    def funcion_coste_tik(self, s_dx, s_dy, Dx, Dy, Z0,Z, lamb):
        '''

        '''
        if self.z is None:
            print("No se ha integrado como para calcular la funcion costo")
            return None
        c_1 = np.linalg.norm(Z @ Dx.T - s_dx, 'fro') ** 2
        c_2 = np.linalg.norm(Dy @ Z - s_dy, 'fro') ** 2
        # regularizacion:
        c_reg = 2*lamb**2 * np.linalg.norm(Z-Z0, 'fro') ** 2

        coste = c_1 + c_2 + c_reg

        print(f'El valor de la funcion coste ha sido de:\n'
              f'Z * Lx^T - Z_x: {c_1} \n'
              f'Z * Ly^T - Z_y: {c_2} \n'
              f'2 * lamb *(Z - Z0):{c_reg} \n'
              f'coste: {coste} \n')
        return coste


    '------------------------CORRECCIONES----------------------'
    def corregir_plano(self,z_reco):
        # z = np.copy(self.z)
        z = np.copy(z_reco)
        # z[:,200:-200]=0

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
        # plano_estimado = np.min(plano_estimado) - plano_estimado

        z_corregido = z - plano_estimado

        # z_corregido = z + plano_estimado
        # self.z = z_corregido
        n,m=z.shape
        # Mostrar resultados
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Original Topography
        cax_0 = axs[0].imshow(z, cmap='plasma', aspect=None,extent=[0, m, 0, n])
        fig.colorbar(cax_0, ax=axs[0], orientation='vertical',shrink=0.55, label=r'Z $(\mu m)$',pad=0.02)
        axs[0].set_title('Topografía Original')
        axs[0].set_xlabel(r'X $(\mu m)$')
        axs[0].set_ylabel(r'Y $(\mu m)$')
        # axs[0].axis('off')

        # Plano Estimado
        cax_1 = axs[1].imshow(plano_estimado, cmap='plasma', aspect=None,extent=[0, m, 0, n])
        fig.colorbar(cax_1, ax=axs[1], orientation='vertical',shrink=0.55, label=r'Z $(\mu m)$',pad=0.02)
        axs[1].set_title('Plano Estimado')
        axs[1].set_xlabel(r'X $(\mu m)$')
        axs[1].set_ylabel(r'Y $(\mu m)$')
        # axs[1].axis('off')

        # Topografía Corregida
        cax_2 = axs[2].imshow(z_corregido, cmap='plasma', aspect=None,extent=[0, m, 0, n])
        fig.colorbar(cax_2, ax=axs[2], orientation='vertical',shrink=0.55, label=r'Z $(\mu m)$',pad=0.02)
        axs[2].set_title('Topografía Corregida')
        axs[2].set_xlabel(r'X $(\mu m)$')
        axs[2].set_ylabel(r'Y $(\mu m)$')
        # axs[2].axis('off')

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
        return z_corregido


    def corregir_polinomio(self, z_reco,verpol=False,grado=3):
        # z = self.z
        z=np.copy(z_reco)
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

        if verpol:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            # cmaps = ['viridis', 'plasma', 'inferno']
            # cmaps=['plasma']

            for ax, data, title in zip(axs, [z, z_estimado, z_corregido],
                                             ['Topografía Original', 'Superficie Polinomial Estimada',
                                              'Topografía Corregida'],
                                       ):
                n,m=z.shape
                img = ax.imshow(data, cmap='plasma', aspect=None,extent=[0, m, 0, n])
                ax.set_title(title)
                ax.set_xlabel(r'X $(\mu m)$')
                ax.set_ylabel(r'Y $(\mu m)$')
                # ax.axis('off')
                fig.colorbar(img, ax=ax,shrink=0.55, label=r'Z $(\mu m)$',pad=0.02)

            plt.show()
            # Métricas
            print('\n Valores correccion desviacion planar (polinomio): \n --------------\n')
            print("Coeficientes del polinomio:", coeficientes)
            print("Suma de residuos cuadrados:", residuals)
            print("Mean Squared Error (MSE):", mse)
            print("Mean Absolute Error (MAE):", mae)
            # self.z = z_corregido
            # return z_corregido, z_estimado, coeficientes, mse, mae
        return z_corregido


    '---------------------------PLOT--------------------------'
    def plot_superficie(self,z, ver_textura=True):
        # plt.ion()
        # matplotlib.use('TkAgg')
        x, y = np.meshgrid(np.arange(self.z.shape[1]), np.arange(self.z.shape[0]))

        sin_textura = plt.figure()
        axis_1 = sin_textura.add_subplot(111, projection='3d')
        axis_1.plot_surface(x * self.dpixel, y * self.dpixel, z, cmap='plasma',shade=True)
        print(x.shape, y.shape)
        print((x * self.dpixel).shape, (y * self.dpixel).shape)
        print(self.dpixel)

        axis_1.set_title('Topografia sin textura')
        axis_1.set_xlabel('X (mm)')
        axis_1.set_ylabel('Y (mm)')
        axis_1.set_zlabel('Z (mm)')

        # axis_1.set_xlabel(r'X $(\mu m)$')
        # axis_1.set_ylabel(r'Y $(\mu m)$')
        # axis_1.set_zlabel(r'Z $(\mu m)$')

        # axis_1.set_xlim([z.shape[1] * self.dpixel, 0])
        # axis_1.set_ylim([0, z.shape[0] * self.dpixel])

        # axis_1.xlim(-self.dpixel, self.dpixel)
        # axis_1.set_zlim(bottom=-40, top=200)
        # tal=[-5,0,5]
        # axis_1.set_zticks(tal)
        #
        axis_1.get_proj = lambda: np.dot(Axes3D.get_proj(axis_1), np.diag([1.0, 1.0, 0.4, 1]))
        axis_1.tick_params(axis='both', which='major', labelsize=7)
        self.z
        mappable = cm.ScalarMappable(cmap=cm.plasma)
        mappable.set_array(z)
        # plt.colorbar(mappable, ax=axis_1, orientation='vertical', label=r'Z $(\mu m)$', shrink=0.5, pad=0.1)
        plt.colorbar(mappable, ax=axis_1, orientation='vertical', label='Z (mm)', shrink=0.5, pad=0.1)
        if ver_textura and self.datos.textura is not None:

            con_textura = plt.figure()
            axis_2 = con_textura.add_subplot(111, projection='3d')

            if self.datos.textura.shape[0] != z.shape[0] or self.datos.textura.shape[1] != z.shape[1]:
                print(f'La forma de la imagen es: {self.datos.textura.shape}')
                print(f'La forma de la funcion es: {z.shape}')
                self.textura = cv2.resize(self.datos.textura, (z.shape[1], z.shape[0]))
                print(
                    'Hemos tenido que reajustar la dimension de la textura por que no coincidia, mira a ver que todo ande bien...')

            axis_2.plot_surface(x * self.dpixel, y * self.dpixel, z, facecolors=self.datos.textura / 255.0, shade=False)

            axis_2.set_title('Topografia con textura')
            axis_2.set_xlabel('X (mm)')
            axis_2.set_ylabel('Y (mm)')
            axis_2.set_zlabel('Z (mm)')

            # axis_2.set_xlabel(r'X $(\mu m)$')
            # axis_2.set_ylabel(r'Y $(\mu m)$')
            # axis_2.set_zlabel(r'Z $(\mu m)$')
            # axis_2.set_zticks(np.arange(-20, 40, 20))

            axis_2.get_proj = lambda: np.dot(Axes3D.get_proj(axis_2), np.diag([1.0, 1.0, 0.5, 1]))

            axis_2.tick_params(axis='both', which='major', labelsize=7)
            # axis_2.secondary_xaxis()
            axis_2.grid(True)
            # axis_2.set_zlim(bottom=-20, top=40)
            # axis_2.set_zlim(-20,200)

            mappable_gray = cm.ScalarMappable(cmap=cm.gray)
            mappable_gray.set_array(z)

            plt.colorbar(mappable_gray, ax=axis_2, orientation='vertical', label='Z (mm)', shrink=0.5, pad=0.1)
        plt.show()


    def mapa_calor(self,z):
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        n, m = z.shape
        topo = ax.imshow(z,cmap='plasma',extent=[0, m, 0, n])
        colorbar= fig.colorbar(topo, ax=ax, cmap='plasma',shrink=0.77, label=r'Z $(\mu m)$',pad=0.02)
        # ax.set_title('Topografía')
        ax.set_xlabel(r'X $(\mu m)$')
        ax.set_ylabel(r'Y $(\mu m)$')
        # ax.axis('off')
        plt.show()


class Contornos:
    def __init__(self, Reconstruccion):
        self.z = Reconstruccion.z
        self.dpixel = Reconstruccion.dpixel


    def parametros3d(self,z):
        # Calcula parámetros de rugosidad
        Sa = np.mean(np.abs(z - np.mean(z)))
        Sq = np.sqrt(np.mean((z - np.mean(z)) ** 2))

        # Aplanamos la matriz para poder utilizar find_peaks
        z_flattened = z.ravel()
        picos, _ = find_peaks(z_flattened)
        valles, _ = find_peaks(-z_flattened)

        # Encuentra los 5 picos más altos y los 5 valles más profundos
        if len(picos) >= 5 and len(valles) >= 5:
            picos_importantes = picos[np.argsort(z_flattened[picos])[-5:]]
            valles_importantes = valles[np.argsort(-z_flattened[valles])[-5:]]
            Sp = np.max(z_flattened[picos_importantes])
            Sv = np.min(z_flattened[valles_importantes])
            Sz = Sp - Sv
            S10z = (np.sum(z_flattened[picos_importantes]) - np.sum(z_flattened[valles_importantes]))/5
        else:
            Sp, Sv, Sz, S10z = np.nan, np.nan, np.nan, np.nan  # En caso de no tener suficientes picos o valles

        # Calcula la asimetría del perfil
        Ssk = (np.sum((z_flattened - np.mean(z_flattened)) ** 3) / len(z_flattened)) / (Sq ** 3)

        # Impresión de resultados
        print('Análisis de Rugosidades para la superficie completa:\n----------------')
        print(f'Sa (Media aritmética de las alturas absolutas): {Sa}')
        print(f'Sq (Raíz cuadrada del promedio de los cuadrados): {Sq}')
        print(f'Sp (Altura máxima del pico): {Sp}')
        print(f'Sv (Profundidad máxima del valle): {Sv}')
        print(f'Sz (Altura máxima): {Sz}')
        print(f'S10z (Suma de las alturas de los 5 picos y valles más prominentes): {S10z}')
        print(f'Ssk (Asimetría): {Ssk}')

        return Sa, Sq, Sp, Sv, Sz, S10z, Ssk
    # def parametros3d(self,z):
    #     # Calcula parámetros de rugosidad
    #     Sa = np.mean(np.abs(z - np.mean(z)))
    #     Sq = np.sqrt(np.mean((z - np.mean(z)) ** 2))
    #
    #     # Usar filtros para encontrar picos y valles
    #     footprint = np.array([[1, 1, 1],
    #                           [1, 0, 1],
    #                           [1, 1, 1]])  # Define la vecindad para la comparación
    #
    #     local_max = maximum_filter(z, footprint=footprint) == z  # Picos locales
    #     local_min = minimum_filter(z, footprint=footprint) == z  # Valles locales
    #
    #     picos_importantes = np.where(local_max)
    #     valles_importantes = np.where(local_min)
    #
    #     # Convertir índices a lineales para seleccionar los top 5
    #     linear_picos = np.ravel_multi_index(picos_importantes, dims=z.shape)
    #     linear_valles = np.ravel_multi_index(valles_importantes, dims=z.shape)
    #
    #     if len(linear_picos) >= 5 and len(linear_valles) >= 5:
    #         sorted_picos = np.argsort(z.ravel()[linear_picos])[-5:]
    #         sorted_valles = np.argsort(-z.ravel()[linear_valles])[-5:]
    #
    #         # Índices de los 5 picos y valles más importantes
    #         picos_importantes = np.unravel_index(linear_picos[sorted_picos], shape=z.shape)
    #         valles_importantes = np.unravel_index(linear_valles[sorted_valles], shape=z.shape)
    #
    #         Sp = np.max(z[picos_importantes])
    #         Sv = np.min(z[valles_importantes])
    #         Sz = Sp - Sv
    #         S10z = (np.sum(z[picos_importantes]) - np.sum(z[valles_importantes]))/5
    #     else:
    #         Sp, Sv, Sz, S10z = np.nan, np.nan, np.nan, np.nan  # En caso de no tener suficientes picos o valles
    #
    #     # Calcula la asimetría del perfil
    #     Ssk = (np.sum((z.ravel() - np.mean(z.ravel())) ** 3) / len(z.ravel())) / (Sq ** 3)
    #
    #     # Impresión de resultados
    #     print('Análisis de Rugosidades para la superficie completa:\n----------------')
    #     print(f'Sa (Media aritmética de las alturas absolutas): {Sa}')
    #     print(f'Sq (Raíz cuadrada del promedio de los cuadrados): {Sq}')
    #     print(f'Sp (Altura máxima del pico): {Sp}')
    #     print(f'Sv (Profundidad máxima del valle): {Sv}')
    #     print(f'Sz (Altura máxima): {Sz}')
    #     print(f'S10z (Suma de las alturas de los 5 picos y valles más prominentes): {S10z}')
    #     print(f'Ssk (Asimetría): {Ssk}')
    def plot_rugosidad3d(self,z):
        fig = plt.figure(figsize=(14, 6))

        # Gráfico 3D de la superficie
        ax1 = fig.add_subplot(121, projection='3d')
        X, Y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
        ax1.plot_surface(X, Y, z, cmap='viridis')
        ax1.set_title('Superficie Topográfica')
        ax1.set_xlabel('X (μm)')
        ax1.set_ylabel('Y (μm)')
        ax1.set_zlabel('Z (μm)')

        # Cálculo de parámetros de rugosidad
        Sa, Sq, Sz, Sp, Sv, S10z, Ssk = self.rugosidad_sulperficial(z)

        # Gráfico de barras de los parámetros de rugosidad
        ax2 = fig.add_subplot(122)
        parameters = ['Sa', 'Sq', 'Sz', 'Sp', 'Sv', 'S10z', 'Ssk']
        values = [Sa, Sq, Sz, Sp, Sv, S10z, Ssk]
        ax2.barh(parameters, values, color='skyblue')
        ax2.set_title('Parámetros de Rugosidad')
        ax2.set_xlabel('Valor (μm)')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def parametros2D(self,z):
        perfil = np.mean(z, axis=1)  # Perfil medio a lo largo de un eje
        l = len(perfil)  # Longitud del perfil

        # Calculando Ra y Rq respecto a la línea base original
        Ra = np.mean(np.abs(perfil - np.mean(perfil)))
        Rq = np.sqrt(np.mean((perfil - np.mean(perfil)) ** 2))

        # Identificación de picos y valles
        picos, _ = find_peaks(perfil,distance=50) #80 de normal ehh
        valles, _ = find_peaks(-perfil,distance=80)

        # Top 5 picos y valles
        picos_importantes = picos[np.argsort(perfil[picos])[-5:]]
        valles_importantes = valles[np.argsort(-perfil[valles])[-5:]]

        # Rp, Rv, Rz, y R10z
        Rp = np.max(perfil[picos_importantes])
        Rv = np.min(perfil[valles_importantes])
        Rz = np.abs(Rp) + np.abs(Rv)
        R10z = np.sum(perfil[picos_importantes] - perfil[valles_importantes]) / 5

        # Asimetría (Rsk)
        Rsk = (np.sum((perfil - np.mean(perfil)) ** 3) / len(perfil)) / (Rq ** 3)

        # widths = np.diff(np.sort(np.concatenate([picos_importantes, valles_importantes])))
        # RSm = np.mean(widths)

        # print('valeesimportantes',valles_importantes)
        # print('picosimportantes',picos_importantes)
        # anchovalle=np.diff(np.sort(valles_importantes[-2:]))
        # anchopico=np.diff(np.sort(picos_importantes))

        anchovalle = np.diff(np.sort(valles))
        anchopico = np.diff(np.sort(picos))
        # print('anchivalle',anchovalle)
        # print('anchopico',anchopico)
        anchovalles=np.mean(anchovalle)
        anchopicos=np.mean(anchopico)

        print('Análisis de Rugosidades para el perfil medio:\n----------------')
        print(f'Ra (Desviación aritmética media): {Ra}')
        print(f'Rq (RMS): {Rq}')
        print(f'Rp (Máximo): {Rp}')
        print(f'Rv (Mínimo): {Rv}')
        print(f'Rz (Altura máxima): {Rz}')
        print(f'R10z (Media de altura de 5 picos y 5 valles): {R10z}')
        print(f'Rsk (Asimetría): {Rsk}')
        print(f'Anchura por valles: {anchovalles}')
        print(f'Anchura por picos: {anchopicos}')

        return perfil, Ra, Rq, Rp, Rv, Rz, R10z, Rsk, picos_importantes, valles_importantes

    def plot_rugo2d(self, z, dpixel=1.9853):
        perfil, Ra, Rq, Rp, Rv, Rz, R10z, Rsk, picos_importantes, valles_importantes = self.parametros2D(z)
        linea_base = np.mean(perfil)
        print('dpixeeeeeeeeel:',dpixel)

        # Aplicar dpixel a las posiciones de x para los picos y valles
        x_picos = picos_importantes * dpixel
        x_valles = valles_importantes * dpixel

        plt.figure(figsize=(15, 6))
        plt.plot(np.arange(len(perfil)) * dpixel, perfil, label='Perfil')  # Aplicar dpixel al eje x del perfil
        plt.scatter(x_picos, perfil[picos_importantes], marker='x', color='red', s=100, label='5 Picos')
        plt.scatter(x_valles, perfil[valles_importantes], marker='o', color='blue', s=100, label='5 Valles')

        # Solo la línea vertical para el pico y valle más significativos
        max_pico = np.argmax(perfil[picos_importantes])
        min_valle = np.argmin(perfil[valles_importantes])
        plt.vlines(x=x_picos[max_pico], ymin=linea_base, ymax=perfil[picos_importantes[max_pico]],
                   color='red', linestyle='--', linewidth=1.5, label=f'Rp = {Rp:.2f}')
        plt.vlines(x=x_valles[min_valle], ymin=linea_base, ymax=perfil[valles_importantes[min_valle]],
                   color='blue', linestyle='--', linewidth=1.5, label=f'Rv = {Rv:.2f}')

        plt.axhline(y=Ra, color='green', linestyle='--', label=f'Ra = {Ra:.2f}')
        plt.axhline(y=Rq, color='purple', linestyle='--', label=f'Rq = {Rq:.2f}')
        plt.axhline(y=linea_base, color='black', linestyle='-', label=f'Media')

        plt.xlabel('Y (µm)', fontsize=18)
        plt.ylabel('Z (µm)', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        plt.grid(True)
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

# #
# img_rutas = {'top': 'imagenes/SENOS1-B.BMP', 'bottom': 'imagenes/SENOS1-T.BMP', 'left': 'imagenes/SENOS1-L.BMP',
#              'right': 'imagenes/SENOS1-R.BMP', 'textura': 'imagenes/SENOS1-S.BMP'}

# img_rutas = {'top': 'imagenes/CIRC1_B.BMP','bottom': 'imagenes/CIRC1_T.BMP','left': 'imagenes/CIRC1_L.BMP','right': 'imagenes/CIRC1_R.BMP','textura': 'imagenes/CIRC1.BMP'}
# img_rutas = {'top': 'imagenes/RUEDA1_T.BMP','bottom': 'imagenes/RUEDA1_B.BMP','left': 'imagenes/RUEDA1_L.BMP','right': 'imagenes/RUEDA1_R.BMP','textura': 'imagenes/RUEDA1_S.BMP'}
# img_rutas = {'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}


# img_rutas = {'top': 'imagenes/4-C-T.BMP', 'bottom': 'imagenes/4-C-B.BMP', 'left': 'imagenes/4-C-L.BMP',
#              'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}

# 'modificando nombres...'

# img_rutas = {'top': 'imagenes/4-C-B.BMP', 'bottom': 'imagenes/4-C-T.BMP', 'left': 'imagenes/4-C-R.BMP',
#              'right': 'imagenes/4-C-L.BMP', 'textura': 'imagenes/4-C-S.BMP'}


# img_rutas = {'top': 'imagenes/6-C-T.BMP', 'bottom': 'imagenes/6-C-B.BMP', 'left': 'imagenes/6-C-L.BMP',
#              'right': 'imagenes/6-C-R.BMP', 'textura': 'imagenes/6-C-S.BMP'}
# img_rutas = {'top': 'imagenes/6M-C-T.BMP', 'bottom': 'imagenes/6M-C-B.BMP', 'left': 'imagenes/6M-C-L.BMP',
#               'right': 'imagenes/6M-C-R.BMP', 'textura': 'imagenes/6M-C-S.BMP'}

# img_rutas = {'top': 'imagenes/CIRC1_T.BMP','bottom': 'imagenes/CIRC1_B.BMP','left': 'imagenes/CIRC1_L.BMP','right': 'imagenes/CIRC1_R.BMP','textura': 'imagenes/CIRC1.BMP'}

img_rutas = {'top': 'imagenes/RUEDA1_T.BMP','bottom': 'imagenes/RUEDA1_B.BMP','left': 'imagenes/RUEDA1_L.BMP','right': 'imagenes/RUEDA1_R.BMP','textura': 'imagenes/RUEDA1_S.BMP'}
# img_rutas = {'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}

'pa arriba'
# img_rutas = {'top': 'calibrado/0_4-B.BMP', 'bottom': 'calibrado/0_4-T.BMP', 'left': 'calibrado/0_4-L.BMP',
#              'right': 'calibrado/0_4-R.BMP', 'textura': 'calibrado/0_4-S.BMP'} #son 960 x 1280 pixeles
'averrr_--> pa abajo'
'la 1 del tfg!!!!!!'
# img_rutas = {'top': 'calibrado/0_4-T.BMP', 'bottom': 'calibrado/0_4-B.BMP', 'left': 'calibrado/0_4-L.BMP',
#              'right': 'calibrado/0_4-R.BMP', 'textura': 'calibrado/0_4-S.BMP'}

# img_rutas = {'top': 'calibrado/04P_T.BMP', 'bottom': 'calibrado/04P_B.BMP', 'left': 'calibrado/04P_L.BMP',
#              'right': 'calibrado/04P_R.BMP', 'textura': 'calibrado/04P_S.BMP'}


# img_rutas = {'top': 'calibrado/0_6-T.BMP', 'bottom': 'calibrado/0_6-B.BMP', 'left': 'calibrado/0_6-L.BMP',
#              'right': 'calibrado/0_6-R.BMP', 'textura': 'calibrado/0_6-S.BMP'}

'se supone que bien, pero hacia arriba...'
# img_rutas = {'top': 'calibrado/0_6-B.BMP', 'bottom': 'calibrado/0_6-T.BMP', 'left': 'calibrado/0_6-L.BMP',
#              'right': 'calibrado/0_6-R.BMP', 'textura': 'calibrado/0_6-S.BMP'}

'hacia abajo'
# img_rutas = {'top': 'calibrado/0_6-B.BMP', 'bottom': 'calibrado/0_6-T.BMP', 'left': 'calibrado/0_6-R.BMP',
#              'right': 'calibrado/0_6-L.BMP', 'textura': 'calibrado/0_6-S.BMP'}

# img_rutas = {'top': 'calibrado/0_6-B.BMP', 'bottom': 'calibrado/0_6-T.BMP', 'left': 'calibrado/0_6-L.BMP',
#              'right': 'calibrado/0_6-R.BMP', 'textura': 'calibrado/0_6-S.BMP'}


# img_rutas = {'top': 'imagenes/4-C-B.BMP', 'bottom': 'imagenes/4-C-T.BMP', 'left': 'imagenes/4-C-L.BMP',
#              'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}

matplotlib.use('TkAgg')
#
#
cargar = Cargarimagenes(img_rutas)
cargar.upload_img(img_rutas)
# histograma = Histograma(cargar)
# procesar = Procesarimagenes(cargar)
# ecualizar = Ecualizacion(cargar)
# ecualizar = Ecualizacion(cargar)
# histograma = Histograma(cargar)
# aplanar = Metodosaplanacion(cargar)

# procesar = Procesarimagenes(cargar)
#
#
reco = Reconstruccion(cargar)
sdx,sdy=reco.calculo_gradientes(1.1112,1)
#
m,n=sdx.shape
Lx=reco.ope_dif2(n)
Ly=reco.ope_dif2(m)
Sx = reco.segundadif(n)
Sy = reco.segundadif(m)
#
#
ztiki,_=reco.reg_tikhonov(sdx,sdy,Lx,Ly,Sx,Sy,lamb=0.1)
# ztiki=reco.corregir_polinomio(ztiki)
reco.plot_superficie(ztiki,True)

# contornear = Contornos(reconstruir)
# perfil=contornear.contornear_y(pos_x=480)
# perfi=contornear.contornear_x(pos_y=640)
# perfi=contornear.contornear_x(pos_y=800)
# perfi=contornear.contornear_x(pos_y=200)
# perfil=contornear.contornear_y(pos_x=0)
# perfil=contornear.contornear_y(pos_x=10)
# perfil=contornear.contornear_y(pos_x=1200)
# perfi=contornear.pilacontornos_x(20)


# # 04 calibrada
# img_rutas = {'top': 'calibrado/0_4-T.BMP', 'bottom': 'calibrado/0_4-B.BMP', 'left': 'calibrado/0_4-L.BMP',
#              'right': 'calibrado/0_4-R.BMP', 'textura': 'calibrado/0_4-S.BMP'}

# 06 q no se calibro...
# img_rutas = {'top': 'calibrado/0_6-T.BMP', 'bottom': 'calibrado/0_6-B.BMP', 'left': 'calibrado/0_6-L.BMP',
#              'right': 'calibrado/0_6-R.BMP', 'textura': 'calibrado/0_6-S.BMP'}


# img_rutas = {'top': 'imagenes/4-C-B.BMP', 'bottom': 'imagenes/4-C-T.BMP', 'left': 'imagenes/4-C-L.BMP',
#              'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}


# img_rutas = {'top': 'imagenes/6-C-B.BMP', 'bottom': 'imagenes/6-C-T.BMP', 'left': 'imagenes/6-C-R.BMP',
#              'right': 'imagenes/6-C-L.BMP', 'textura': 'imagenes/6-C-S.BMP'}
# img_rutas = {'top': 'imagenes/6M-C-B.BMP', 'bottom': 'imagenes/6M-C-T.BMP', 'left': 'imagenes/6M-C-L.BMP',
#               'right': 'imagenes/6M-C-R.BMP', 'textura': 'imagenes/6M-C-S.BMP'}

'''
matplotlib.use('TkAgg')

img_rutas = {'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}

'lo normal vaya ajajasajaj'

cargar = Cargarimagenes(img_rutas)
cargar.upload_img(img_rutas)
# histograma = Histograma(cargar)
reco= Reconstruccion(cargar)
sdx,sdy = reco.calculo_gradientes(1.1112,1,eps=1e-5, ver=False)

m,n=sdx.shape
Lx=reco.ope_dif2(n)
Ly=reco.ope_dif2(m)
Sx = reco.segundadif(n)
Sy = reco.segundadif(m)


ztiki,_=reco.reg_tikhonov(sdx,sdy,Lx,Ly,Sx,Sy,lamb=0.1)
ztiki=reco.corregir_polinomio(ztiki)


min=np.min(ztiki)
min_index=np.unravel_index(np.argmin(ztiki),ztiki.shape)
max=np.max(ztiki)
max_index=np.unravel_index(np.argmax(ztiki),ztiki.shape)
print("El valor mínimo es:", min)
print("El índice del valor mínimo es:", min_index)
print("El valor máximo es:", max)
print("El índice del valor máximo es:", max_index)
print('El valor en el borde es:',ztiki[0,0])


reco.plot_superficie(ztiki,False)

contornear = Contornos(reco)
contornear.parametros3d(ztiki)
contornear.plot_rugo2d(ztiki,dpixel=0.7940)
'''

'-----------------------------------------------------------'
'------------------------RUGOSIDADES------------------------'
'-----------------------------------------------------------'

# matplotlib.use('TkAgg')
'''
def comparar_rugosidades(superficies, nombres, contorneadores):
    parametros_labels = ['Ra', 'Rq', 'Rp', '|Rv|', 'Rz', 'R10z']
    datos = {nombre: [] for nombre in nombres}

    for superficie, contornear, nombre in zip(superficies, contorneadores, nombres):
        # contornear.plot_rugo2d(superficie)
        _, Ra, Rq, Rp, Rv, Rz, R10z, _, _, _ = contornear.parametros2D(superficie)
        datos[nombre] = [Ra, Rq, Rp, abs(Rv), Rz, R10z]

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.25
    index = np.arange(len(parametros_labels))*1.1

    for i, nombre in enumerate(nombres):
        bars = ax.bar(index + i * bar_width, datos[nombre], bar_width, label=f'{nombre}')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom',ha='center', fontsize=12)

    ax.set_xlabel('Parámetros de Rugosidad', fontsize=24)
    ax.set_ylabel('Valores', fontsize=20)
    # ax.set_title('Comparación de Rugosidad para Diferentes Superficies', fontsize=25)
    ax.set_xticks(index + bar_width / 2 * (len(nombres) - 1))
    ax.set_xticklabels(parametros_labels, fontsize=17)
    ax.legend(fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=17)

    plt.tight_layout()
    plt.show()
def comparar_rugosidades_3d(superficies, nombres, contorneadores):
    parametros_labels = ['Sa', 'Sq', 'Sp', '|Sv|', 'Sz', 'S10z']
    datos = {nombre: [] for nombre in nombres}

    for superficie, contornear, nombre in zip(superficies, contorneadores, nombres):
        # reco.plot_superficie(superficie,False) #con los 4 diccionarios no funsiona, revisar broadcasting
        Sa, Sq, Sp, Sv, Sz, S10z, _ = contornear.parametros3d(superficie)
        datos[nombre] = [Sa, Sq, Sp, abs(Sv), Sz, S10z]

    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.25
    index = np.arange(len(parametros_labels))*1.1

    for i, nombre in enumerate(nombres):
        bars = ax.bar(index + i * bar_width, datos[nombre], bar_width, label=f'{nombre}')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center', fontsize=12)

    ax.set_xlabel('Parámetros de Rugosidad', fontsize=20)
    ax.set_ylabel('Valores', fontsize=20)
    # ax.set_title('Comparación de Rugosidad 3D para Diferentes Superficies', fontsize=25)
    ax.set_xticks(index + bar_width / 2 * (len(nombres) - 1))
    ax.set_xticklabels(parametros_labels, fontsize=17)
    ax.legend(fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=17)

    plt.tight_layout()
    plt.show()

# Procesar cada conjunto de imágenes y comparar
img_rutas = {
    '04mm': {'top': 'imagenes/4-C-B.BMP', 'bottom': 'imagenes/4-C-T.BMP', 'left': 'imagenes/4-C-L.BMP',
             'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'},
    '06mm': {'top': 'imagenes/6-C-B.BMP', 'bottom': 'imagenes/6-C-T.BMP', 'left': 'imagenes/6-C-R.BMP',
             'right': 'imagenes/6-C-L.BMP', 'textura': 'imagenes/6-C-S.BMP'},
    '06mm_mec': {'top': 'imagenes/6M-C-B.BMP', 'bottom': 'imagenes/6M-C-T.BMP', 'left': 'imagenes/6M-C-L.BMP',
                 'right': 'imagenes/6M-C-R.BMP', 'textura': 'imagenes/6M-C-S.BMP'},
    'BJ':{'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}
}

superficies = []
contorneadores = []
nombres = ['0.4mm', '0.6mm', '0.6mm Mecanizada','BJ']

for key, rutas in img_rutas.items():
    cargar = Cargarimagenes(rutas)
    cargar.upload_img(rutas)
    proseso=Procesarimagenes(cargar)
    proseso.filtro(5,False)
    reco = Reconstruccion(cargar)
    sdx, sdy = reco.calculo_gradientes(1.1112, 1, eps=1e-5, ver=False)
    z, _ = reco.reg_tikhonov(sdx, sdy, reco.ope_dif2(sdx.shape[1]), reco.ope_dif2(sdy.shape[0]),
                             reco.segundadif(sdx.shape[1]), reco.segundadif(sdy.shape[0]), lamb=0.1)
    z = reco.corregir_polinomio(z)
    contornear = Contornos(reco)
    superficies.append(z)
    contorneadores.append(contornear)

comparar_rugosidades(superficies, nombres, contorneadores)
comparar_rugosidades_3d(superficies, nombres,contorneadores)
'''

'-----------------------------------------------------------'
'-------------MONTECARLO; comparación de tiempos------------'
'-----------------------------------------------------------'


'''
print('Iniciando test comparativo de tiempos entre la solucion aplicando la transformación de Householder y sin aplicarla')

def prueba_syl(sdx,sdy,Lx_s, Ly_s):
    start_time = time.time()
    Z_s = self.solve_sylvester(sdx,sdy,Lx_s, Ly_s)
    print(f"Duración de la ejecución: {time.time() - start_time} segundos")


def prueba_hou(sdx,sdy,Lx2,Ly2):
    start_time = time.time()
    Z_h = self.householder(sdx,sdy,Lx2, Ly2)
    print(f"Duración de la ejecución: {time.time() - start_time} segundos")


cargar = Cargarimagenes(img_rutas)
procesar=Procesarimagenes(cargar)

cargar.upload_img(img_rutas)
self = Reconstruccion(cargar)
sdxorigi,sdyorigi = self.calculo_gradientes(1,1,eps=1e-5, ver=False)


amplitud_sdx = np.max(sdxorigi) - np.min(sdxorigi)
amplitud_sdy = np.max(sdyorigi) - np.min(sdyorigi)
amplitud_gradientes = (amplitud_sdx + amplitud_sdy) / 2

sigmas = np.linspace(0, 0.1 * amplitud_gradientes*255, 10)
varianzas = sigmas**2
# varianzas_porcentaje = (varianzas / varianza_gradientes) * 100
# varianzas_porcentaje=np.arange(0,11,1)
varianzas_porcentaje=np.linspace(0,10,10)

print(amplitud_sdx,amplitud_sdy)
print('amplitud',amplitud_gradientes)
print(f"Valores de sigma: {sigmas}")
print('varianzas:',varianzas)
print('varianzas_porcentaje',varianzas_porcentaje)

#varibales globlales:
m=960
n=1280
objeto = Reconstruccion(cargar)
Lx_s= objeto.ope_dif2(n)
Ly_s= objeto.ope_dif2(m)

Lx2 = self.ope_dif2(n)
Ly2 = self.ope_dif2(m)


# tiempo = timeit.repeat('prueba_syl()', setup='from __main__ import prueba_syl', number=3, repeat=10)
# print(f"Tiempo promedio de ejecución de sylvester: {sum(tiempo) / (len(tiempo)*3)} segundos")
#
# tiempo = timeit.repeat('prueba_hou()', setup='from __main__ import prueba_hou', number=3, repeat=10)
# print(f"Tiempo promedio de ejecución de householder: {sum(tiempo) / (len(tiempo)*3)} segundos")

time_s=[]
time_h=[]

for sigma in sigmas:
    cargar.add_gaussian_noise(0,sigma=sigma)
    procesar.nivel_ruido()
    sdx,sdy=self.calculo_gradientes(1, 1, eps=1e-5, ver=False)

    tiempo_syl = timeit.repeat(lambda: prueba_syl(sdx, sdy, Lx_s, Ly_s), number=1, repeat=10)
    tiempo_hou = timeit.repeat(lambda: prueba_hou(sdx, sdy, Lx2, Ly2), number=1, repeat=10)

    # Calcula el promedio de los tiempos medidos
    ts = sum(tiempo_syl) / len(tiempo_syl)
    th = sum(tiempo_hou) / len(tiempo_hou)

    time_s.append(ts)
    time_h.append(th)

    cargar.upload_img(img_rutas)

plt.figure()
plt.plot(varianzas_porcentaje, time_s, label='Sylvester')
plt.plot(varianzas_porcentaje, time_h, label='Householder')
plt.xlabel(r'$\sigma^2$')
plt.ylabel('Tiempo (s)')
plt.title('Tiempo de Ejecución vs. Nivel de ruido')
plt.legend()
plt.grid(True)
plt.show()


tiempos_syl = np.array(time_s)  # Suponiendo que time_s contiene todos los tiempos de Sylvester
tiempos_hou = np.array(time_h)  # Suponiendo que time_h contiene todos los tiempos de Householder

print("Estadísticas Sylvester:")
print("Promedio:", np.mean(tiempos_syl))
print("Mediana:", np.median(tiempos_syl))
print("Desviación estándar:", np.std(tiempos_syl))
print("Mínimo:", np.min(tiempos_syl), "Máximo:", np.max(tiempos_syl))

print("\nEstadísticas Householder:")
print("Promedio:", np.mean(tiempos_hou))
print("Mediana:", np.median(tiempos_hou))
print("Desviación estándar:", np.std(tiempos_hou))
print("Mínimo:", np.min(tiempos_hou), "Máximo:", np.max(tiempos_hou))

plt.hist(tiempos_syl, bins=10, alpha=0.7, label='Sylvester')
plt.hist(tiempos_hou, bins=10, alpha=0.7, label='Householder')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia')
plt.title('Histograma de Tiempos de Ejecución')
plt.legend()
plt.show()

# Boxplot
plt.boxplot([tiempos_syl, tiempos_hou], labels=['Sylvester', 'Householder'])
plt.ylabel('Tiempo (s)')
plt.title('Boxplot de Tiempos de Ejecución')
plt.show()

# Gráfico de línea de tiempos vs. ruido
plt.plot(varianzas_porcentaje, tiempos_syl, label='Sylvester')
plt.plot(varianzas_porcentaje, tiempos_hou, label='Householder')
plt.xlabel('Porcentaje de Varianza del Ruido')
plt.ylabel('Tiempo (s)')
plt.title('Tiempo de Ejecución vs. Nivel de ruido')
plt.legend()
plt.grid(True)
plt.show()

from scipy import stats

# Prueba T (asumiendo normalidad)
t_stat, p_value = stats.ttest_ind(tiempos_syl, tiempos_hou)
print("Prueba T:", "T-Statistic:", t_stat, "P-Value:", p_value)

# Prueba U de Mann-Whitney (no paramétrica)
u_stat, p_value = stats.mannwhitneyu(tiempos_syl, tiempos_hou)
print("Prueba U de Mann-Whitney:", "U-Statistic:", u_stat, "P-Value:", p_value)
'''


'-------------------------------------------------------------------'
'--------------------Metodo L-Curve para tikhonov-------------------'
'-------------------------------------------------------------------'
'''
cargar = Cargarimagenes(img_rutas)
reconstruir = Reconstruccion(cargar)

m=960
n=1280

Lx, _ = reconstruir.diffmat2(n - 1, (0, n - 1))
Ly, _ = reconstruir.diffmat2(m - 1, (0, m - 1))

Sx = reconstruir.segundadif(n)
Sy = reconstruir.segundadif(m)
# Z_t = self.reg_tikhonov(Lx, Ly, Sx, Sy, lamb=1e-2)


def calculate_residuals_and_solution_norm(A, B, C, Z):
    residual_norm = np.linalg.norm(A @ Z + Z @ B + C)
    solution_norm = np.linalg.norm(Z)
    print('tiene q ser lo mismo q error tiki:', A @ Z + Z @ B + C)
    print('residual norm: ', residual_norm)
    print('solution norm: ', solution_norm)
    return residual_norm, solution_norm

# lambdas = np.logspace(-6, 2, 7)  # Valores de lambda desde 10^-4 hasta 10^1
# lambdas=[1e-3,1e-2,1e-1,2e-1,3e-1]
# lambdas = np.logspace(-6, 0, 7)
lambdas=[1e-6 ,1e-5 ,1e-4, 1e-3, 1e-2, 1e-1,2e-1, 1e0]
# lambdas=[0]
# lambdas = np.logspace(-6, 1, 50)
residuals = []
solutions = []
print('lambdas: ', lambdas)
for lamb in lambdas:
    Z,A,B,C = reconstruir.reg_tikhonov(Lx, Ly, Sx, Sy, lamb)
    res_norm, sol_norm = calculate_residuals_and_solution_norm(A, B, C, Z)
    residuals.append(res_norm)
    solutions.append(sol_norm)


plt.figure()
plt.loglog(residuals, solutions, marker='o')
for i in range(len(lambdas)):
    plt.loglog(residuals[i],solutions[i],'o', label=f'λ={lambdas[i]}')

plt.xlabel('Norma de los Residuos')
plt.ylabel('Norma de la Solución')
plt.grid()
plt.legend(loc='best')
plt.show()

residuals = np.array(residuals)  
solutions = np.array(solutions)  
'''




'-------------------------------------------------------------------'
'--------------------Metodo De Montecarlo-------------------'
'-------------------------------------------------------------------'
'''
def error_reconstruccion(Z_reconstruida, Z_real):
    numerador = np.linalg.norm(Z_reconstruida - Z_real)
    denominador = np.linalg.norm(Z_real)
    return numerador / denominador

def calcular_residuo_sistema(error):
    """
    Calcula el residuo del sistema Az + zB = C para la matriz Z dada.
    """
    residuo = np.linalg.norm(error)
    return residuo


def funcion_coste(Z, Dx, Dy, Sx, Sy,Sxorigi,Syorigi):
    # Calcular los términos de los gradientes
    error_x = Z @ Dx.T - Sx
    error_y = Dy @ Z - Sy
    termino_x = np.linalg.norm(error_x, 'fro') ** 2 / np.linalg.norm(Sxorigi, 'fro') ** 2
    termino_y = np.linalg.norm(error_y, 'fro') ** 2 / np.linalg.norm(Syorigi, 'fro') ** 2

    # Calcular la función de coste total
    coste = termino_x + termino_y

    return coste


def coste_tikhonov(Z, Dx, Dy, Sx, Sy, Lx, Ly,Sxorigi,Syorigi, lamb):
    error_x =  Z @ Dx.T - Sx
    error_y = Dy @ Z - Sy
    termino_x = np.linalg.norm(error_x, 'fro') ** 2 / np.linalg.norm(Sxorigi, 'fro') ** 2
    termino_y = np.linalg.norm(error_y, 'fro') ** 2 / np.linalg.norm(Syorigi, 'fro') ** 2
    reg_x = np.linalg.norm(Z @ Lx.T, 'fro') ** 2 / np.linalg.norm(Lx, 'fro') ** 2
    reg_y = np.linalg.norm(Ly @ Z, 'fro') ** 2 / np.linalg.norm(Ly, 'fro') ** 2
    coste = termino_x + termino_y + lamb * (reg_x + reg_y)
    return coste

cargar = Cargarimagenes(img_rutas)
procesar=Procesarimagenes(cargar)

cargar.upload_img(img_rutas)
self = Reconstruccion(cargar)
sdxorigi,sdyorigi = self.calculo_gradientes(1,1,eps=1e-5, ver=False)
# sdxorigi=self.s_dx
# sdyorigi=self.s_dy

m=960
n=1280

Lx_s = self.ope_dif2(n)
Ly_s = self.ope_dif2(m)

Lx2 = self.ope_dif2(n)
Ly2 = self.ope_dif2(m)

Lx, _ = self.diffmat2(n - 1, (0, n - 1))
Ly, _ = self.diffmat2(m - 1, (0, m - 1))

Sx = self.segundadif(n)
Sy = self.segundadif(m)

Lx_d = self.ope_dif2(n)
Ly_d = self.ope_dif2(m)


Z_real_s,_,_ = self.solve_sylvester(sdxorigi,sdyorigi,Lx_s, Ly_s)
Z_real_h,_,_ = self.householder(sdxorigi,sdyorigi,Lx2, Ly2)
Z_real_t,_ = self.reg_tikhonov(sdxorigi,sdyorigi,Lx, Ly, Sx, Sy, lamb=1e-1)
Z_real_d,_ = self.reg_dirichlet(sdxorigi,sdyorigi,Lx_d, Ly_d)


amplitud_sdx = np.max(sdxorigi) - np.min(sdxorigi)
amplitud_sdy = np.max(sdyorigi) - np.min(sdyorigi)
amplitud_gradientes = (amplitud_sdx + amplitud_sdy) / 2

sigmas = np.linspace(0, 0.4 * amplitud_gradientes*255, 40)
varianzas = sigmas**2
# varianzas_porcentaje = (varianzas / varianza_gradientes) * 100
# varianzas_porcentaje=np.arange(0,11,1)
varianzas_porcentaje=np.linspace(0,40,40)

print(amplitud_sdx,amplitud_sdy)
print('amplitud',amplitud_gradientes)
print(f"Valores de sigma: {sigmas}")
print('varianzas:',varianzas)
print('varianzas_porcentaje',varianzas_porcentaje)


# sigmas = np.linspace(0, 100, 10)  # De 0 a 10

# sigmas = np.linspace(0, 10, 11)

# sigmas = np.linspace(0, 20, 21)
# varianzas = sigmas**2
# print(sigmas)

comparacion_zs=[]
comparacion_zh=[]
comparacion_zt=[]
comparacion_zd=[]

R_ZS=[]
R_ZH=[]
R_ZT=[]
R_ZD=[]

coste_s=[]
coste_h=[]
coste_t=[]
coste_d=[]

time_s=[]
time_h=[]

for sigma in sigmas:
    cargar.add_gaussian_noise(0,sigma=sigma)
    procesar.nivel_ruido()
    sdx,sdy=self.calculo_gradientes(1, 1, eps=1e-5, ver=False)
    # sdx = self.s_dx
    # sdy = self.s_dy

    Z_s, es,ts = self.solve_sylvester(sdx,sdy,Lx_s, Ly_s)
    Z_h, eh,th = self.householder(sdx,sdy,Lx2, Ly2)
    Z_t, et = self.reg_tikhonov(sdx,sdy,Lx, Ly, Sx, Sy, lamb=1e-1)
    Z_d, ed = self.reg_dirichlet(sdx,sdy,Lx_d, Ly_d)

    R_ZS.append(calcular_residuo_sistema(es))
    R_ZH.append(calcular_residuo_sistema(eh))
    R_ZT.append(calcular_residuo_sistema(et))
    R_ZD.append(calcular_residuo_sistema(ed))

    comparacion_zs.append(error_reconstruccion(Z_s,Z_real_s))
    comparacion_zh.append(error_reconstruccion(Z_h,Z_real_h))
    comparacion_zt.append(error_reconstruccion(Z_t,Z_real_t))
    comparacion_zd.append(error_reconstruccion(Z_d,Z_real_d))

    coste_s.append(funcion_coste(Z_s,Lx_s,Ly_s,sdx,sdy,sdxorigi,sdyorigi))
    coste_h.append(funcion_coste(Z_h,Lx2,Ly2,sdx,sdy,sdxorigi,sdyorigi))
    coste_t.append(coste_tikhonov(Z_t, Lx, Ly,sdx,sdy, Sx, Sy,sdxorigi,sdyorigi, lamb=1e-1))
    coste_d.append(funcion_coste(Z_d,Lx_d,Ly_d,sdx,sdy,sdxorigi,sdyorigi))

    time_s.append(ts)
    time_h.append(th)

    cargar.upload_img(img_rutas)

    print(R_ZS)
    print(comparacion_zs)
    print(coste_s)


plt.figure()
plt.plot(varianzas_porcentaje, R_ZS, label='Sylvester')
plt.plot(varianzas_porcentaje, R_ZH, label='Householder')
plt.plot(varianzas_porcentaje, R_ZT, label='Tikhonov')
plt.plot(varianzas_porcentaje, R_ZD, label='Dirichlet')
plt.xlabel(r'$\sigma^2$ (%)')
plt.ylabel(r'Residuo $\epsilon$')
plt.title('Residuo del sistema vs. Nivel de ruido')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(varianzas_porcentaje, comparacion_zs, label='Sylvester')
plt.plot(varianzas_porcentaje, comparacion_zh, label='Householder')
plt.plot(varianzas_porcentaje, comparacion_zt, label='Tikhonov')
plt.plot(varianzas_porcentaje, comparacion_zd, label='Dirichlet')
plt.xlabel(r'$\sigma^2$ (%)')
plt.ylabel('Error Relativo (%)')
plt.title('Error Relativo de Reconstrucción vs. Nivel de ruido')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(varianzas_porcentaje, coste_s, label='Sylvester')
plt.plot(varianzas_porcentaje, coste_h, label='Householder')
plt.plot(varianzas_porcentaje, coste_t, label='Tikhonov')
plt.plot(varianzas_porcentaje, coste_d, label='Dirichlet')
plt.xlabel(r'$\sigma^2$ (%)')
plt.ylabel(r'$\epsilon$')
plt.title('Función de Coste vs. Nivel de ruido')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(varianzas_porcentaje, time_s, label='Sylvester')
plt.plot(varianzas_porcentaje, time_h, label='Householder')
plt.xlabel(r'$\sigma^2$')
plt.ylabel('Tiempo (s)')
plt.title('Tiempo de Ejecución vs. Nivel de ruido')
plt.legend()
plt.grid(True)
plt.show()
'''


'--------------------------------------------------'
'-----------------------RPSDF----------------------'
'--------------------------------------------------'
'''
cargar = Cargarimagenes(img_rutas)
# procesar=Procesarimagenes(cargar)
cargar.upload_img(img_rutas)

self = Reconstruccion(cargar)
sdx,sdy = self.calculo_gradientes(1,1,eps=1e-5, ver=False)

m=960
n=1280

Lx_s = self.ope_dif2(n)
Ly_s = self.ope_dif2(m)

Lx2 = self.ope_dif2(n)
Ly2 = self.ope_dif2(m)

Lx, _ = self.diffmat2(n - 1, (0, n - 1))
Ly, _ = self.diffmat2(m - 1, (0, m - 1))

Sx = self.segundadif(n)
Sy = self.segundadif(m)

Lx_d = self.ope_dif2(n)
Ly_d = self.ope_dif2(m)



Z_s,_,_ = self.solve_sylvester(sdx,sdy,Lx_s, Ly_s)
Z_h,_,_ = self.householder(sdx,sdy,Lx2, Ly2)
Z_d,_ = self.reg_dirichlet(sdx,sdy,Lx_d, Ly_d)
Z_t,_ = self.reg_tikhonov(sdx,sdy,Lx, Ly, Sx, Sy, lamb=1e-1)

Z_pol=self.corregir_polinomio(Z_s)
Z_pla=self.corregir_plano(Z_s)



z_list = [Z_s, Z_h, Z_d, Z_t, Z_pol, Z_pla]
z_name = ['No regularizada', 'Householder', 'Dirichlet', 'Tikhonov', 'Polinomio', 'Plano']


self.plot_rpsd(z_list,z_name)


'Para diferentes \lambdassss'

Z_t21,_ = self.reg_tikhonov(sdx,sdy,Lx, Ly, Sx, Sy, lamb=2e-1)
Z_t11,_ = self.reg_tikhonov(sdx,sdy,Lx, Ly, Sx, Sy, lamb=1e-1)
Z_t12,_ = self.reg_tikhonov(sdx,sdy,Lx, Ly, Sx, Sy, lamb=1e-2)
Z_t13,_ = self.reg_tikhonov(sdx,sdy,Lx, Ly, Sx, Sy, lamb=1e-3)
Z_t14,_ = self.reg_tikhonov(sdx,sdy,Lx, Ly, Sx, Sy, lamb=1e-4)

z_list=[Z_t21, Z_t11,Z_t12, Z_t13,Z_t14]
z_name = ['λ=0.2','λ=0.1','λ=0.01','λ=0.001','λ=0.0001']

self.plot_rpsd(z_list,z_name)
'''
'-------------------------------------------------'
'-------------Pequeñas comprobaciones-------------'
'-------------------------------------------------'

'''
cargar = Cargarimagenes(img_rutas)
cargar.upload_img(img_rutas)
reco= Reconstruccion(cargar)
n=100

opedif2=reco.ope_dif2(n,1)
segundadif=reco.segundadif(n)

Ldifmat,Sdifmat=reco.diffmat2(n-1,(0, n - 1))


print('opedif2...\n',opedif2)
print('Ldifmat...\n',Ldifmat)

print('segundadif...\n',segundadif)
print('Sdifmat...\n',Sdifmat)

mmm=np.allclose(opedif2,Ldifmat,atol=1e-5)
print(mmm)

mmm2=np.allclose(segundadif,Sdifmat)
print(mmm2)
'''



end_total=time.time()
print(f'Se ha ejecutado todo el código en un tiempo de: {end_total-start_total}')