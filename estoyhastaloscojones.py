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
from scipy.linalg import lstsq
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import linalg
import time
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
        self.aplicar_fourier(ver=False)
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

    def filtro(self, sigma = 20, ver = True):
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
            r=50
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
        # self.dpixel = 1 / 251 # piezas bj (primera sesion)
        # self.dpixel = 500/251 # piezas FDM (15 abril)
        self.dpixel = 0.9598 #pixels/um #calibracion
        # calibracion (04 --> a'=a*0.853) ! 06 -->

        'Gradientes --> siempre activa'
        self.calculo_gradientes(1,1,eps=1e-5, ver=False) #c=85.36, d=100


        'metodos de integracion'
        self.least_squares()


        "Reconstrucción y visualizador 3D"
        self.plot_superficie(ver_textura=True)


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
    def ope_difforward(self,n, h=1):

        D = np.zeros((n, n))

        for i in range(0, n):
            D[i, i] = -1
            D[i - 1, i] = 1

        D[-1, - 1] = 1
        D[-1, 0] = 0
        D[-1, -2] = -1

        D = D / h
        return D


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


    def ope_dif3(self,n, h=1):
        D = np.zeros((n, n))

        for i in range(0, n):
            D[i - 1, i] = 1
            D[i, i - 1] = -1

        D[-1, 0] = 0
        D[0, -1] = 0
        D[0, 0] = -3
        D[-1, -1] = 3
        D[0, 1] = 4
        D[-1, -2] = -4
        D[0, 2] = -1
        D[-1, -3] = 1

        D = D / (2 * h)
        return D


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

    def diffcheb(self,n, xspan):
        """
        Compute Chebyshev differentiation matrices on `n`+1 points in the
        interval `xspan`. Returns a vector of nodes and the matrices for the
        first and second derivatives.
        """
        # Nodes in [-1,1]
        x = -np.cos(np.arange(n + 1) * np.pi / n)

        # Off-diagonal entries
        c = np.hstack(([2], np.ones(n - 1), [2]))  # endpoint factors
        D_x = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    D_x[i, j] = (-1) ** (i + j) * c[i] / (c[j] * (x[i] - x[j]))

        # Diagonal entries
        D_x[np.isinf(D_x)] = 0  # fix divisions by zero on diagonal
        s = np.sum(D_x, axis=1)
        np.fill_diagonal(D_x, -s)  # "negative sum trick"

        # Transplant to [a, b]
        a, b = xspan
        x = a + (b - a) * (x + 1) / 2
        D_x = 2 * D_x / (b - a)  # chain rule

        # Second derivative
        D_xx = np.dot(D_x, D_x)

        return  D_x, D_xx

    '-----------------METODOS DE INTEGRACION-----------------'
    def householder(self,Lx,Ly):
        '''

        :param Lx:
        :param Ly:
        :return:
        '''
        print('\n Integracion mediante el método HOUSEHOLDER \n --------------------------- \n')
        # m, n = self.s_dx.shape

        n=Lx.shape[0] #ambas son nxn o mxm
        m=Ly.shape[0]
        self.todo_mat(Lx, 'Lx',False)
        self.todo_mat(Ly, 'Ly',False)

        #Matrices ortogonales, sirven de apoyo
        Px= self.calc_P(n)
        Py = self.calc_P(m)
        self.orto(Px,'Px')
        self.orto(Py,'Py')

        #caluamos L gorro --> primer elemento nulo y condicionamiento menor
        Lxg = Lx @ Px
        Lyg = Ly @ Py
        self.todo_mat(Lxg,'Lxg',False)
        self.todo_mat(Lyg,'Lyg',False)

        #imponemos que sean 0 los lugares que son aproximadamente (mejora muy levemente el codnicionamiento)
        tol = 1e-5
        Lxg[np.abs(Lxg) < tol] = 0
        Lyg[np.abs(Lyg) < tol] = 0

        #Calulamos los terminos del sistema de lyapuvnov
        lytly = Lyg.T @ Lyg
        lxtlx = Lxg.T @ Lxg

        #termino independiente (Q)
        G1 = Lyg.T @ self.s_dy @ Px
        G2 = Py.T @ self.s_dx @ Lxg
        G = -(G1 + G2)

        #Extraemos las submatrices para resolver el sistema lineal (bibliografia)
        A = lytly[1:, 1:]
        B = lxtlx[1:, 1:]
        c10 = G[1:, 0]
        c01 = G[0, 1:]
        C = G[1:, 1:]

        self.cond(A,'A')
        self.cond(B,'B')

        # A y B int¡vertibles, el sistema es lineal

        w01 = np.linalg.solve(B, -c01.T)
        print('sistema w01:')
        print(np.allclose(np.dot(w01.T, B) + c01, 0, atol=1e-10))  # Deberia retornar True si esta correcto

        print('sistema w10:')
        w10 = np.linalg.solve(A, -c10)
        print(np.allclose(np.dot(w10, A) + c10, 0, atol=1e-10))

        #Se usó en linalg.lstsq y .solve y dio mejores resultados .solve a pesar del cond A y B
        # error_w01 = w01.T @ B + c01.T
        # error_w10 = A @ w10 + c10

        #resolvemos el termino W11 segun la nueva formulacion Sylvester:
        W11 = solve_sylvester(A, B, -C)
        error = A @ W11 + W11 @ B + C
        print("Error de la solución Lyapunov:", error)  #muy bajo, es 0

        #reconstruimos W... W[0,0]=0 sería como Z0, nos d aigual
        W = np.zeros((len(w10) + 1, len(w01) + 1))
        W[0, 1:] = w01
        W[1:, 0] = w10
        W[1:, 1:] = W11

        Px_inv = np.linalg.inv(Px.T)
        Py_inv_T = np.linalg.inv(Py)

        #Resolvemos Z
        Z = Py @ W @ Px.T
        self.z=Z
        return Z


    def hou_tikho(self,Lx,Ly,lamb=1e-1):
        '''
        Funcion que resuelve la ecuacion de Sylvester para la integracion de los gradientes
        ayudandose de la reformulacion de housholder y añadiendo la regularizacion de
        tikchonov. Es identica a housholder excepto en las matrices A y B
        :param Lx: matriz operador diferenciacion de dimencion nxn
        :param Ly: matriz operador diferenciacion de dimencion mxm
        :return: Z superficie reconstruida
        '''
        n = Lx.shape[0]  # ambas son nxn o mxm
        m = Ly.shape[0]
        # self.cond(Lx, 'Lx')
        # self.plot_cond(Lx,'Lx')
        # self.orto(Lx,'Lx')
        # self.cond(Ly, 'Ly')
        # self.plot_cond(Ly,'Ly')
        # self.orto(Ly,'Ly')
        self.todo_mat(Lx, 'Lx', False)
        self.todo_mat(Ly, 'Ly', False)

        # Matrices ortogonales, sirven de apoyo
        Px = self.calc_P(n)
        Py = self.calc_P(m)

        self.orto(Px, 'Px')
        self.orto(Py, 'Py')

        # caluamos L gorro --> primer elemento nulo y condicionamiento menor
        Lxg = Lx @ Px
        Lyg = Ly @ Py

        self.todo_mat(Lxg, 'Lxg', False)
        self.todo_mat(Lyg, 'Lyg', False)

        # imponemos que sean 0 los lugares que son aproximadamente
        tol = 1e-5
        Lxg[np.abs(Lxg) < tol] = 0
        Lyg[np.abs(Lyg) < tol] = 0

        # Calulamos los terminos del sistema de lyapuvnov
        lytly = Lyg.T @ Lyg
        lxtlx = Lxg.T @ Lxg

        # termino independiente (Q)
        G1 = Lyg.T @ self.s_dy @ Px
        G2 = Py.T @ self.s_dx @ Lxg
        G = -(G1 + G2)

        # Extraemos las submatrices para resolver el sistema lineal(bibliografia)
        A = lytly[1:, 1:]
        B = lxtlx[1:, 1:]
        c10 = G[1:, 0]
        c01 = G[0, 1:]
        C = G[1:, 1:]

        self.cond(A, 'A')
        self.cond(B, 'B')

        # A y B int¡vertibles, el sistema es lineal

        w01 = np.linalg.solve(B, -c01.T)
        print('sistema w01:')
        print(np.allclose(np.dot(w01.T, B) + c01, 0, atol=1e-10))  # Deberia retornar True si esta correcto

        print('sistema w10:')
        w10 = np.linalg.solve(A, -c10)
        print(np.allclose(np.dot(w10, A) + c10, 0, atol=1e-10))

        # Se usó en linalg.lstsq y .solve y dio mejores resultados .solve a pesar del cond A y B
        # error_w01 = w01.T @ B + c01.T
        # error_w10 = A @ w10 + c10
        'Regularizacion de tikhinov'

        At = A + lamb * np.eye(A.shape[0])
        Bt = B+ lamb * np.eye(B.shape[0])

        # resolvemos el termino W11 segun la nueva formulacion Sylvester:
        W11 = solve_sylvester(At, Bt, -C)
        error = At @ W11 + W11 @ B + C
        print("Error de la solución Lyapunov:", error)  # muy bajo, es 0

        # reconstruimos W... W[0,0]=0 sería como Z0, nos d aigual
        W = np.zeros((len(w10) + 1, len(w01) + 1))
        W[0, 1:] = w01
        W[1:, 0] = w10
        W[1:, 1:] = W11

        Px_inv = np.linalg.inv(Px.T)
        Py_inv_T = np.linalg.inv(Py)

        # Resolvemos Z
        Z = Py @ W @ Px.T
        self.z = Z
        return Z


    def solve_sylvester(self,Lx,Ly):

        A = Ly.T @ Ly
        B = Lx.T @ Lx
        C = - (Ly.T @ self.s_dy + self.s_dx @ Lx)

        Z = solve_sylvester(A, B, -C)
        error = A @ Z + Z @ B + C
        print("Error de la solución Lyapunov:", error)


    def reg_dirichlet(self,Lx,Ly):
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
        C = - P.T @ (Ly.T @ self.s_dy + self.s_dx @ Lx) @ Q

        print(A)
        print(B)
        print(C)

        Z = solve_sylvester(A, B, -C)
        error = A @ Z + Z @ B + C
        print("Error de la solución Lyapunov:", error)
        self.z = Z
        return Z


    def reg_tikhonov(self,Lx,Ly,Sx,Sy,lamb=1e-2):
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
        # Sy= Ly @ Ly
        # Sx= Lx @ Lx
        # print(Lx.shape, Ly.shape, Sx.shape, Sy.shape)
        # Z0 = self.reg_dirichlet(Lx, Ly)
        # Z0=np.ones((Ly.shape[1],Lx.shape[0]))
        Z0 = np.zeros((Ly.shape[1], Lx.shape[0]))
        # print(Z0.shape)
        # print(Sy.T @ Sy)
        # print(Sx.T @ Sx)
        A= Ly.T @ Ly + lamb**2 * Sy.T @ Sy
        B = Lx.T @ Lx + lamb**2 * Sx.T @ Sx
        # A = Ly.T @ Ly + lamb ** 2 * np.eye(Ly.shape[0])
        # B = Lx.T @ Lx + lamb ** 2 * np.eye(Lx.shape[0])

        C = -((Ly.T @ self.s_dy + self.s_dx @ Lx) + lamb**2 * (Sy.T @ Sy @ Z0 + Z0 @ Sx.T @ Sx))
        tol=1e-10
        # A[np.abs(A) < tol] =0
        # C[np.abs(C) < tol] = 0
        # B[np.abs(B) < tol] = 0
        # print(A)
        # print(B)
        # print(C)
        Z = solve_sylvester(A, B, -C)

        error = A @ Z + Z @ B + C
        print("Error de la solución Lyapunov mediante tikhonov:\n", error)
        self.z=Z
        # self.funcion_coste_tik(self.s_dx,self.s_dy,Lx,Ly,Z0,Z,lamb)

    def reg_tikhonov2(self,Lx,Ly,lamb=1e-2):
        print('Aplicando tikhonov...')
        Ux,Sx,Vxt = np.linalg.svd(Lx, full_matrices=False)
        Uy, Sy, Vyt = np.linalg.svd(Ly, full_matrices=False)

        P=Uy@self.s_dy@Vxt.T
        Q=Vyt@self.s_dx@Ux
        Sy=np.diag(Sy)
        Sx = np.diag(Sx)
        # print(Sx)

        A=Sy@Sy+(lamb**2)*np.eye(Sy.shape[0])
        B=Sx@Sx+(lamb**2)*np.eye(Sx.shape[0])
        print('sy',Sy.shape)
        print('p', P.shape)
        print('q', Q.shape)
        print('sx', Sx.shape)
        # x=Sy.T@Sy
        # print(Q.dtype)
        C=-(Sy@P+Q@Sx)

        Z=solve_sylvester(A, B,-C)
        error = A @ Z + Z @ B + C
        print("Error de la solución Lyapunov mediante tikhonov:\n", error)
        self.z = Z




    '-----------------------INTEGRACION POR MINIMOS CUADRADOS-----------------'
    def least_squares(self):
        m, n = self.s_dx.shape
        '----------------------------SYLVESTER------------------------------'
        # Lx_s = self.ope_dif3(n)
        # Ly_s = self.ope_dif3(m)

        # Lx_s = self.ope_dif2(n)
        # Ly_s = self.ope_dif2(m)
        # Z_s= self.solve_sylvester(Lx_s,Ly_s)
        # Z_t=self.reg_tikhonov2(Lx_s,Ly_s,Z_s,1e-1)
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
        # Z_house_2 = self.householder(Lx2, Ly2)
        # Z_t=self.reg_tikhonov2(Lx2,Ly2,Z_house_2,1e-2)
        # coste_hou_2 = self.funcion_coste(self.s_dx, self.s_dy, Lx2, Ly2, Z_house_2)
        # self.funcion_coste_tik(self.s_dx, self.s_dy, Lx2, Ly2, Z_house_2,Z_t,1e-1)

        # Lx3 = self.ope_dif3(n)
        # Ly3 = self.ope_dif3(m)
        # Z_house_3=self.householder(Lx3,Ly3)
        # coste_hou=self.funcion_coste(self.s_dx,self.s_dy,Lx3,Ly3,Z_house_3)



        '-------------------TIKHONOV----------------------'
        Lx = self.ope_dif2(n)
        Ly = self.ope_dif2(m)
        # Sx=Lx@Lx
        # Sy=Ly@Ly
        # Lx = self.ope_dif3(n)
        # Ly = self.ope_dif3(m)

        # Lx,Sx = self.diffmat2(n-1,(0,n-1))
        # Ly,Sy = self.diffmat2(m-1,(0,m-1))
        # Lx, Sx = self.diffcheb(n - 1, (0, n - 1))
        # Ly, Sy = self.diffcheb(m - 1, (0, m - 1))
        #

        #
        # self.reg_tikhonov(Lx,Ly,Sx,Sy,lamb=1e-1)
        # self.reg_tikhonov2(Lx,Ly,lamb=1e-1)
        # self.corregir_polinomio()
        # self.corregir_plano()


        '------------------DIRITCHLET------------------'

        # Lx_s = self.ope_dif2(n)
        # Ly_s = self.ope_dif2(m)

        # Lx_s, Sx = self.diffcheb(n - 1, (0, n - 1))
        # Ly_s, Sy = self.diffcheb(m - 1, (0, m - 1))
        # Z_s = self.reg_dirichlet(Lx_s, Ly_s)





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
    def corregir_plano(self):
        z = np.copy(self.z)
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


    def corregir_polinomio(self, grado=4):
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
        self.z = z_corregido
        # return z_corregido, z_estimado, coeficientes, mse, mae


    '---------------------------PLLOT--------------------------'
    def plot_superficie(self, ver_textura=True):
        # plt.ion()
        matplotlib.use('TkAgg')
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

        # axis_1.get_proj = lambda: np.dot(Axes3D.get_proj(axis_1), np.diag([1.0, 1.0, 0.4, 1]))
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

            # axis_2.get_proj = lambda: np.dot(Axes3D.get_proj(axis_2), np.diag([1.0, 1.0, 0.4, 1]))

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

# #
# img_rutas = {'top': 'imagenes/SENOS1-B.BMP', 'bottom': 'imagenes/SENOS1-T.BMP', 'left': 'imagenes/SENOS1-L.BMP',
#              'right': 'imagenes/SENOS1-R.BMP', 'textura': 'imagenes/SENOS1-S.BMP'}

# img_rutas = {'top': 'imagenes/CIRC1_T.BMP','bottom': 'imagenes/CIRC1_B.BMP','left': 'imagenes/CIRC1_L.BMP','right': 'imagenes/CIRC1_R.BMP','textura': 'imagenes/CIRC1.BMP'}

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

# img_rutas = {'top': 'imagenes/RUEDA1_T.BMP','bottom': 'imagenes/RUEDA1_B.BMP','left': 'imagenes/RUEDA1_L.BMP','right': 'imagenes/RUEDA1_R.BMP','textura': 'imagenes/RUEDA1_S.BMP'}
# img_rutas = {'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}

#
img_rutas = {'top': 'calibrado/0_4-B.BMP', 'bottom': 'calibrado/0_4-T.BMP', 'left': 'calibrado/0_4-L.BMP',
             'right': 'calibrado/0_4-R.BMP', 'textura': 'calibrado/0_4-S.BMP'} #son 960 x 1280 pixeles


# img_rutas = {'top': 'calibrado/0_6-T.BMP', 'bottom': 'calibrado/0_6-B.BMP', 'left': 'calibrado/0_6-L.BMP',
#              'right': 'calibrado/0_6-R.BMP', 'textura': 'calibrado/0_6-S.BMP'}

'se supone que bien, pero hacia arriba...'
# img_rutas = {'top': 'calibrado/0_6-B.BMP', 'bottom': 'calibrado/0_6-T.BMP', 'left': 'calibrado/0_6-L.BMP',
#              'right': 'calibrado/0_6-R.BMP', 'textura': 'calibrado/0_6-S.BMP'}

'hacia abajo'
# img_rutas = {'top': 'calibrado/0_6T.BMP', 'bottom': 'calibrado/0_6-B.BMP', 'left': 'calibrado/0_6-R.BMP',
#              'right': 'calibrado/0_-6-L.BMP', 'textura': 'calibrado/0_6-S.BMP'}


# img_rutas = {'top': 'imagenes/4-C-B.BMP', 'bottom': 'imagenes/4-C-T.BMP', 'left': 'imagenes/4-C-L.BMP',
#              'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}




cargar = Cargarimagenes(img_rutas)
# histograma = Histograma(cargar)
# procesar = Procesarimagenes(cargar)
# ecualizar = Ecualizacion(cargar)
# ecualizar = Ecualizacion(cargar)
# histograma = Histograma(cargar)
# aplanar = Metodosaplanacion(cargar)

# procesar = Procesarimagenes(cargar)
#
#
reconstruir = Reconstruccion(cargar)

# contornear = Contornos(reconstruir)
# perfil=contornear.contornear_y(pos_x=480)
# perfi=contornear.contornear_x(pos_y=640)
# perfi=contornear.contornear_x(pos_y=800)
# perfi=contornear.contornear_x(pos_y=200)
# perfil=contornear.contornear_y(pos_x=0)
# perfil=contornear.contornear_y(pos_x=10)
# perfil=contornear.contornear_y(pos_x=1200)
# perfi=contornear.pilacontornos_x(20)