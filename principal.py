# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:55:10 2024

@author: Jorge
"""
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import cv2
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import lstsq

from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class reco_superficie3d:
    def __init__(self,img_rutas,dpixel=1/251.8750):
        #le damos atributos de variables locales a nuestro objeto --> self
        
        self.img_ruta=img_rutas
        self.img_dict={}
        self.z=None # --> z es un atributo de nuestro objeto que calcularemos mas tarde, así nos curamos en salud con posibles errores si intentamos ver z antes de calcularlo
        self.dpixel=dpixel
        self.textura=None #lo mismo
        self.ruido=None

        'aplicamos nuestras cositass'
        self.upload_imagenes()  # Cargar las imágenes inmediatamente
        # self.histogrameando()
        
        'filtros'
        # self.aplicar_ruido()
        # self.aplicar_filtro(ver=False)
        
        'transformada'
        # self.aplicar_fourier(ver=False)
        self.ecualizar()
        # self.aplanacion()
        # self.aplicar_fourier(ver=True)
        # self.aplicar_filtro(ver=True)
        # self.ecualizar()
        self.histogrameando()
        
    'Filtro gaussiano'
    def ver_ruido(self,image):
        ruido = np.std(image)
        print(f'el nivel de ruido es de: {ruido}')
        return ruido

    def aplicar_ver_ruido(self):
        for key, image in self.img_dict.items():
            if key != 'textura':
                self.ruido(image)
                
    def aplicar_filtro(self, sigma=1,ver=True):
        
        for key, image in self.img_dict.items():        
            if key != 'textura':
                img_no_filtrada=self.img_dict[key]
                self.img_dict[key]=gaussian_filter(image,sigma=sigma)
                
                if ver==True:
                    canva=plt.figure(figsize=(8,3))
                    original=canva.add_subplot(121)
                    original.imshow(img_no_filtrada,cmap='gray')
                    original.set_title(f'Original: {key}')
                    original.axis('off')
                    
                    filtrada=canva.add_subplot(122)
                    filtrada.imshow(self.img_dict[key],cmap='gray')
                    filtrada.set_title(f'Filtrada: {key}')
                    filtrada.axis('off')
                    
                    canva.tight_layout()
                    canva.show()
        
    'Transformada de fourier'
    def aplicar_transformada_fourier(self, image):
        """
        Aplica la Transformada de Fourier a la imagen y retorna el espectro de frecuencia desplazado.
        """
        f_transform = fft2(image)  # Calcula la FFT de la imagen.
        f_shift = fftshift(
            f_transform)  # Desplaza el resultado para que el componente de baja frecuencia esté en el centro.
        return f_shift

    def estimar_radio_filtro(self, f_shift):
        """
        Estima el valor del radio para el filtro pasa-bajas en base al espectro de frecuencia.
        Por ahora, lo estableceremos de forma fija, pero puedes implementar una lógica para determinarlo automáticamente.
        """
        # Esta es una estimación simple y deberías ajustarla según las necesidades de tu análisis.
        # Por ejemplo, podrías encontrar el radio donde la energía cae por debajo de un umbral dado.
        r = 250
        return r

    def aplicar_filtro_y_transformada_inversa(self, f_shift, r):
        """
        Aplica un filtro pasa-bajas y luego la Transformada de Fourier Inversa.
        """
        rows, cols = f_shift.shape
        crow, ccol = rows // 2, cols // 2

        # Crear un filtro pasa-bajas circular.
        mask = np.zeros((rows, cols), np.uint8)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r ** 2
        mask[mask_area] = 1

        # Aplicar la máscara y la transformada inversa.
        f_shift_masked = f_shift * mask
        f_ishift = ifftshift(f_shift_masked)
        image_back = np.abs(ifft2(f_ishift))
        return image_back

    def aplicar_fourier(self,ver=True):
        """
        Orquesta el procesamiento de Fourier y el filtrado, sobrescribiendo las imágenes en el diccionario.
        """
        for key, image in self.img_dict.items():
            if key != 'textura':  # No procesar la textura
                f_shift = self.aplicar_transformada_fourier(image)
                r = self.estimar_radio_filtro(f_shift)
                image_back = self.aplicar_filtro_y_transformada_inversa(f_shift, r)

                # Calcular y mostrar métricas de calidad de la imagen.
                ssim_value = ssim(image, image_back, data_range=image.max() - image.min())
                psnr_value = psnr(image, image_back, data_range=image.max() - image.min())
                print(f"{key} - SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f}")

                if ver ==True:
                    # Visualizar las imágenes originales y filtradas.

                    plt.figure(figsize=(10, 4))
                    plt.title(f'{key} - SSIM: {ssim_value:.4f}, PSNR: {psnr_value:}')
                    plt.subplot(1, 2, 1)
                    plt.imshow(image, cmap='gray')
                    plt.title(f'Original: {key}')
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(image_back, cmap='gray')
                    plt.title(f'Filtrada: {key}')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.show()

                # Sobrescribir la imagen en el diccionario.
                self.img_dict[key] = image_back

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
                # print(image.dtype)
                if image is None:
                    raise ValueError(f'La imagen {key} no se pudo cargar. Mira a ver que estén bien las rutas...')
   
            # self.img_dict[key]=image.astype(np.float32) #Vamos a necesitar coma flotante 32 bits para no perder informacion (cv.im... lee en 8 bits)
            self.img_dict[key]=image

    'metodo de aplanacion mediante minimos cuadrdados'
    def obtener_numero_condicion(self, matriz_sist,ver=True):
        numero_condicion = np.linalg.cond(matriz_sist)
        print(f"El número de condición de la matriz es: {numero_condicion}")
        if ver==True:
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

        num_cond = self.obtener_numero_condicion(matriz_sist,ver=False)

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
        
        # print(s_dx)
        # print(s_dy)
        
        # Acumulación a lo largo de axis=1 --> x / axis=0 -->y
        # z_x=cumtrapz(s_dx*c/d, dx=self.dpixel, axis=1, initial=z0)
        # z_y=cumtrapz(s_dy*c/d, dx=self.dpixel, axis=0, initial=z0)
        z_x = cumulative_trapezoid(s_dx * c / d, dx=self.dpixel, axis=1, initial=z0)
        z_y = cumulative_trapezoid(s_dy * c / d, dx=self.dpixel, axis=0, initial=z0)
        
        self.z=z_x+z_y #ahora self.z ya no es None
        # print(self.z)
        # print(np.max(self.z))
        # print(np.min(self.z))
        # self.z=self.z/(np.max(self.z))
        # print(np.max(self.z))
        # print(np.min(self.z))        
    def plot_superficie(self, ver_textura=True):     
        # plt.ion()
        
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
    
    # def ecualizar(self):
    #     for key, image in self.img_dict.items():
    #         #imagen en escala gris?
    #         if image.ndim == 2 or image.shape[2] == 1: #el shape[2] es si a imagen esta en escala RGB
    #             print(image.dtype)
    #
    #             #cambiara si hago esto¿?
    #             if image.dtype != np.uint8:
    #                 print(image.dtype)
    #                 print('algo raro hayyy eh')
    #
    #                 image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #                 image = image.astype(np.uint8)
    #
    #             # Ecualizar la imagen
    #             self.img_dict[key] = cv2.equalizeHist(image)
    #         else:
    #             print(f"La imagen {key} no ta en escala de grises y no se puede ecuañlizar.")

    def calcular_contraste(self, image):
        return np.std(image)

    def calcular_entropia(self, image):
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
        hist_norm = hist / hist.sum()
        entropia = -np.sum(
            hist_norm * np.log2(hist_norm + np.finfo(float).eps))  # np.finfo(float).eps para evitar log(0)
        return entropia

    def ecualizar(self):
        for key, image in self.img_dict.items():
            # Verificar si la imagen está en escala de grises
            if image.ndim == 2 or image.shape[2] == 1:
                print(f"Procesando imagen {key}")

                # Calcular contraste y entropía originales
                contraste_original = self.calcular_contraste(image)
                entropia_original = self.calcular_entropia(image)

                # Normalización previa si es necesario
                if image.dtype != np.uint8:
                    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    image = image.astype(np.uint8)

                # Aplicar CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image_ecualizada = clahe.apply(image)

                # Calcular contraste y entropía después de la ecualización
                contraste_ecualizado = self.calcular_contraste(image_ecualizada)
                entropia_ecualizada = self.calcular_entropia(image_ecualizada)

                # Actualizar la imagen en el diccionario
                self.img_dict[key] = image_ecualizada

                # Imprimir mejoras
                print(f"Imagen {key} - Mejora de Contraste: {contraste_ecualizado - contraste_original}")
                print(f"Imagen {key} - Mejora de Entropía: {entropia_ecualizada - entropia_original}")
            else:
                print(f"La imagen {key} no está en escala de grises y no se puede ecualizar.")

# img_rutas = {'top': 'imagenes/SENOS1-T.BMP','bottom': 'imagenes/SENOS1-B.BMP','left': 'imagenes/SENOS1-L.BMP','right': 'imagenes/SENOS1-R.BMP','textura': 'imagenes/SENOS1-S.BMP'}
img_rutas = {'top': 'imagenes/CIRC1_T.BMP','bottom': 'imagenes/CIRC1_B.BMP','left': 'imagenes/CIRC1_L.BMP','right': 'imagenes/CIRC1_R.BMP','textura': 'imagenes/CIRC1.BMP'}

# img_rutas = {'top': 'imagenes/RUEDA1_T.BMP','bottom': 'imagenes/RUEDA1_B.BMP','left': 'imagenes/RUEDA1_L.BMP','right': 'imagenes/RUEDA1_R.BMP','textura': 'imagenes/RUEDA1_S.BMP'}
# img_rutas = {'top': 'imagenes/RUEDA3_T.BMP','bottom': 'imagenes/RUEDA3_B.BMP','left': 'imagenes/RUEDA3_L.BMP','right': 'imagenes/RUEDA3_R.BMP','textura': 'imagenes/RUEDA3.BMP'}

mi_superficie = reco_superficie3d(img_rutas)

mi_superficie.integracion(c=1, d=1, z0=0)


mi_superficie.plot_superficie(ver_textura=True)

# mi_superficie.contornear_x(20)
    

# mi_superficie.pilacontornos_x(20)
            
        
'''
Verificación del Proceso de Aplanamiento: asegurarme de que
 el aplanamiento elimina correctamente las inclinaciones de la 
 imagen sin afectar la altura real de las características q me importan de verdad --> como hago esto?
'''
        
        