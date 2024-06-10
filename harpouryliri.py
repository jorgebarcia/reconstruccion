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
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
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
        # self.aplicar_filtro_pasa_bajas()
        # self.aplicar_correccion_planar()

    def aplicar_filtro_pasa_bajas(self, sigma=50, ver=False):
        '''
        Aplica un filtro gaussiano a nuestras imágenes para suavizarlas.
        '''
        print("\nAplicando filtro pasa bajas...\n")
        img_dict_filtrada = {}  # Diccionario auxiliar para las imágenes filtradas
        for key, image in self.datos.img_dict.items():
            if key != 'textura':  # No aplicar a la textura
                img_filtrada = gaussian_filter(image, sigma=sigma)
                img_dict_filtrada[key + '_filtrada'] = img_filtrada  # Guardar las imágenes filtradas

                if ver:
                    plt.figure(figsize=(8, 4))
                    plt.subplot(121)
                    plt.imshow(image, cmap='gray')
                    plt.title(f'Original {key}')
                    plt.axis('off')

                    plt.subplot(122)
                    plt.imshow(img_filtrada, cmap='gray')
                    plt.title(f'Filtrada {key}')
                    plt.axis('off')
                    plt.show()

        # Actualizar el diccionario original con las imágenes filtradas
        self.datos.img_dict.update(img_dict_filtrada)

    def aplicar_correccion_planar(self):
        '''
        Aplica corrección planar a las imágenes usando matrices de coeficientes correctores.
        '''
        print("\nAplicando corrección de desviación planar...\n")

        # Cálculo de las intensidades iniciales en el centro (0,0) de las imágenes filtradas
        intensidades_centro = {key: image[0, 0] for key, image in self.datos.img_dict.items() if '_filtrada' in key}
        suma_intensidades_centro = sum(intensidades_centro.values())

        img_dict_corregida = {}  # Diccionario auxiliar para las imágenes corregidas
        for key, image in self.datos.img_dict.items():
            if 'filtrada' not in key and key != 'textura':  # No corregir la textura ni las imágenes ya filtradas
                imagen_filtrada = self.datos.img_dict[key + '_filtrada']
                intensidad_centro = intensidades_centro[key + '_filtrada']

                # Calcular la matriz de coeficientes de corrección
                kappa_i = (intensidad_centro / (imagen_filtrada + np.finfo(float).eps)) * (
                            suma_intensidades_centro / (4 * intensidad_centro))

                # Aplicar la corrección a la imagen original
                imagen_corregida = image * kappa_i

                # Guardar la imagen corregida en el diccionario auxiliar
                img_dict_corregida[key + '_corregida'] = imagen_corregida.astype(np.uint8)

                print(f'Corrección aplicada a {key}')

                # Visualización opcional
                plt.figure(figsize=(8, 4))
                plt.subplot(121)
                plt.imshow(image, cmap='gray')
                plt.title(f'Original {key}')
                plt.axis('off')

                plt.subplot(122)
                plt.imshow(imagen_corregida, cmap='gray')
                plt.title(f'Corregida {key}')
                plt.axis('off')
                plt.show()

        # Actualizar el diccionario original con las imágenes corregidas
        self.datos.img_dict.update(img_dict_corregida)

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
    def filtro(self, sigma=5, ver=True):
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

        # self.solve_lyapunov_equation()
        # self.integracion_bidireccional()

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
        # s_dx = (i_a - i_b) / np.clip(i_a + i_b, eps, np.inf)
        # s_dy =(i_d - i_c) / np.clip(i_c + i_d, eps, np.inf)

        # c_A = 1.0
        # c_B = 1.0
        # d_A = 1.0
        # d_B = 1.0
        #
        #
        #
        # self.s_dx= (c_A + c_B) / (d_A + d_B) * S_AB + (c_B - c_A) / (d_A + d_B) * (1 - (d_A - d_B) / (d_A + d_B) * S_AB)

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

    # def create_difference_operator(self, n):
    #     e = np.ones(n)
    #     D = diags([-e, e], [0, 1], shape=(n, n)).toarray()
    #     D[-1, -1] = 0  # Ajustar la última fila
    #     return D
    #
    # def solve_lyapunov_equation(self):
    #     n, m = self.s_dx.shape
    #
    #     # Crear operadores de diferenciación con diferencias finitas centradas
    #     Dx = self.create_difference_operator(m)
    #     Dy = self.create_difference_operator(n)
    #
    #     # Construir matrices laplacianas
    #     Lx = Dx.T @ Dx
    #     Ly = Dy.T @ Dy
    #
    #     # Regularización de Tikhonov (parámetro de suavizado)
    #     lambda_reg = 1e-5
    #     I_nm = eye(n * m)
    #
    #     # Crear la matriz combinada con regularización
    #     Laplacian = (kron(eye(m), Ly) + kron(Lx, eye(n)))**2 + (lambda_reg * I_nm)**2
    #
    #     # Calcular los términos de la ecuación de Lyapunov en ambas direcciones
    #     Gx = self.s_dx @ Dx.T
    #     Gy = Dy @ self.s_dy
    #
    #     # Sumar los gradientes en ambas direcciones
    #     G_combined = Gx + Gy
    #
    #     # Vectorizar el gradiente combinado
    #     b = G_combined.flatten()
    #
    #     # Resolver la ecuación usando un solucionador para matrices dispersas
    #     Z_flat = spsolve(Laplacian, b)
    #     Z = Z_flat.reshape(n, m)
    #     self.z=Z
    #     return Z

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
#              'right': 'calibrado/0_4-R.BMP', 'textura': 'calibrado/0_4-S.BMP'} #son 960 x 1280 pixeles


# img_rutas = {'top': 'calibrado/0_6-T.BMP', 'bottom': 'calibrado/0_6-B.BMP', 'left': 'calibrado/0_6-L.BMP',
#              'right': 'calibrado/0_6-R.BMP', 'textura': 'calibrado/0_6-S.BMP'}

'se supone que bien, pero hacia arriba...'
# img_rutas = {'top': 'calibrado/0_6-B.BMP', 'bottom': 'calibrado/0_6-T.BMP', 'left': 'calibrado/0_6-L.BMP',
#              'right': 'calibrado/0_6-R.BMP', 'textura': 'calibrado/0_6-S.BMP'}

'hacia abajo'
img_rutas = {'top': 'calibrado/0_6-T.BMP', 'bottom': 'calibrado/0_6-B.BMP', 'left': 'calibrado/0_6-R.BMP',
             'right': 'calibrado/0_6-L.BMP', 'textura': 'calibrado/0_6-S.BMP'}


# img_rutas = {'top': 'imagenes/4-C-B.BMP', 'bottom': 'imagenes/4-C-T.BMP', 'left': 'imagenes/4-C-L.BMP',
#              'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}




cargar = Cargarimagenes(img_rutas)
# histograma = Histograma(cargar)
# procesar = Procesarimagenes(cargar)
# ecualizar = Ecualizacion(cargar)
# histograma = Histograma(cargar)
# aplanar = Metodosaplanacion(cargar)

procesar = Procesarimagenes(cargar)
#
#
reconstruir = Reconstruccion(cargar)

# contornear = Contornos(reconstruir)
# perfil=contornear.contornear_y(pos_x=480)
# perfi=contornear.contornear_x(pos_y=640)
# # perfi=contornear.contornear_x(pos_y=200)
# perfil=contornear.contornear_y(pos_x=0)
# perfil=contornear.contornear_y(pos_x=10)
# perfil=contornear.contornear_y(pos_x=1200)
# perfi=contornear.pilacontornos_x(20)
