# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:56:15 2024

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

class reco_superficie3d:
    def __init__(self,img_rutas,dpixel=10/251.8750):
        #le damos atributos de variables locales a nuestro objeto --> self
        
        self.img_ruta=img_rutas
        self.img_dict={}
        self.z=None # --> z es un atributo de nuestro objeto que calcularemos mas tarde, así nos curamos en salud con posibles errores si intentamos ver z antes de calcularlo
        self.dpixel=dpixel
        self.textura=None #lo mismo
        self.upload_imagenes()  # Cargar las imágenes inmediatamente
        # self.histogrameando()
        
        # self.ecualizar()
        # self.transformar()
        # self.aplicar_ransac_todas()
        self.aplanacion()
        
        # self.aplanacion_spline_evaluar()
        
        
        self.histogrameando()

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
    'ransac'
    def aplicar_ransac_todas(self):
        for key, image in self.img_dict.items():
            if key != 'textura':  # Excluir la imagen de textura
                try:
                    self.img_dict[key], _ = self.ransac_aplanar(image, degree=1)
                except Exception as e:
                    print(f"Error al aplicar RANSAC a la imagen {key}: {e}")
                    raise

    def ransac_aplanar(self, image, degree=1, max_trials=1000, min_samples=0.5, residual_threshold=5.0):
        # Preparar los datos para el ajuste
        index_y, index_x = np.indices(image.shape)
        coords = np.column_stack((index_x.ravel(), index_y.ravel()))
        z = image.ravel()
    
        # Asegúrate de que los datos son válidos
        if np.isnan(coords).any() or np.isnan(z).any():
            raise ValueError("Los datos contienen NaNs.")
        if np.isinf(coords).any() or np.isinf(z).any():
            raise ValueError("Los datos contienen infinitos.")
    
        # Inicializar RANSACRegressor correctamente
        try:
            if degree == 1:
                # Para un ajuste lineal, usamos LinearRegression directamente
                ransac = RANSACRegressor(LinearRegression(), max_trials=max_trials, min_samples=min_samples, 
                                         residual_threshold=residual_threshold, random_state=0)
            else:
                # Para un ajuste polinomial, usamos make_pipeline para incluir PolynomialFeatures
                ransac = RANSACRegressor(make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression()),
                                         max_trials=max_trials, min_samples=min_samples, 
                                         residual_threshold=residual_threshold, random_state=0)
            ransac.fit(coords, z)
        except Exception as e:
            print(f"Error durante el ajuste de RANSAC: {e}")
            raise
    
        # Verificar que el modelo ha sido ajustado
        if not hasattr(ransac, 'inlier_mask_') or ransac.inlier_mask_ is None:
            raise ValueError("RANSAC no encontró suficientes inliers o no se ajustó correctamente.")
    
        # Predicción de z usando el modelo ajustado
        z_estimated = ransac.predict(coords).reshape(image.shape)
    
        # Calcular la imagen aplanada
        image_corrected = image - z_estimated
    
        return image_corrected, ransac.inlier_mask_.reshape(image.shape)
    
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
    
    'spline'
    def evaluar_ajuste_spline(self, image, image_corrected):
        # Calcular residuos
        residuos = image - image_corrected
        
        # Calcular la norma de los residuos
        norma_residual = np.linalg.norm(residuos)
        print(f"Norma de los residuos: {norma_residual}")
        
        # Visualización de los residuos
        plt.figure(figsize=(10, 5))
        
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
        
        return norma_residual
    
    def spline_aplanar(self, image, kx=3, ky=3, puntos_control=None):
        nrows, ncols = image.shape
        
        # Genera puntos de grilla para la imagen completa
        y = np.linspace(0, nrows - 1, nrows)
        x = np.linspace(0, ncols - 1, ncols)
        X, Y = np.meshgrid(x, y)
        
        # Si se especifican puntos de control, crea una grilla reducida para el ajuste
        if puntos_control is not None:
            x_control = np.linspace(0, ncols - 1, puntos_control)
            y_control = np.linspace(0, nrows - 1, puntos_control)
        else:
            x_control = x
            y_control = y
        
        # Crea el spline utilizando la grilla reducida
        spline = RectBivariateSpline(y_control, x_control, image[:puntos_control, :puntos_control], kx=kx, ky=ky)
        
        # Evalúa el spline en la grilla completa
        z_spline = spline.ev(Y.ravel(), X.ravel()).reshape((nrows, ncols))
        
        # Resta el spline evaluado de la imagen original para obtener la imagen corregida
        image_corrected = image - z_spline
        
        return image_corrected

    def aplanacion_spline_evaluar(self):
        for key, image in self.img_dict.items():
            if key != 'textura':  # Excluir la imagen de textura
                # Aplana la imagen utilizando el ajuste spline
                image_corrected = self.spline_aplanar(image)
                # image_corrected = self.spline_aplanar(image, kx=2, ky=2, suavizado=1)
                # image_corrected = self.spline_aplanar(image, kx=3, ky=3, puntos_control=20)
                # Guarda la imagen corregida en el diccionario
                self.img_dict[key] = image_corrected
                # Evalúa el ajuste spline y visualiza los residuos
                self.evaluar_ajuste_spline(image, image_corrected)
    
    'polinomio sklearn'
    def polinomio_aplanar(self, image, degree=2):
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
    
    'transformada'
    
    def transformar(self):
        for key, image in self.img_dict.items():
            if key != 'textura':
                self.img_dict[key]=self.aplicar_transformada_fourier(image)
    
    def aplicar_transformada_fourier(self, image):
        # Aplicar la FFT a la imagen
        f_transform = fft2(image)
        
        # Desplazar el cero de las frecuencias al centro
        f_shift = fftshift(f_transform)
        
        # Aquí aplicarías algún tipo de filtro de frecuencias, por ejemplo:
        # Un filtro pasa-bajas podría ser una máscara circular centrada en el centro de la imagen
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2  # centro
        mask = np.zeros((rows, cols), np.uint8)
        r = 30  # El radio del filtro
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1
        
        # Aplicar la máscara y la transformada inversa de Fourier
        f_shift = f_shift * mask
        f_ishift = ifftshift(f_shift)
        img_back = ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        return img_back
    
    'pincho'
    def preprocesar_y_evaluar(self):
        for key, image in self.img_dict.items():
            if key != 'textura':  # Excluir la imagen de textura
                # Preprocesar para eliminar ruido
                image_suave = gaussian_filter(image, sigma=1)
                
                # Ajuste y evaluación
                image_corrected, residuos, r2 = self.ajustar_y_evaluar(image_suave)
                
                # Almacenar los resultados
                self.img_dict[key] = image_corrected
                print(f'Imagen {key}:')
                print(f' - Norma de los residuos: {np.linalg.norm(residuos)}')
                print(f' - R2: {r2}')
                
                # Visualización
                self.visualizar_ajuste(image, image_corrected, residuos)

    def ajustar_y_evaluar(self, image):
        # Ajuste aquí con el modelo que elijas, p. ej., Ridge con características polinómicas
        X, Y = np.indices(image.shape)
        features = np.column_stack((X.ravel(), Y.ravel()))
        poly = PolynomialFeatures(degree=2, include_bias=False)
        features_poly = poly.fit_transform(features)

        # Ajuste del modelo
        model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=100))
        model.fit(features_poly, image.ravel())

        # Predicción y cálculo de los residuos
        image_pred = model.predict(features_poly).reshape(image.shape)
        residuos = image - image_pred

        # Cálculo de R2
        r2 = r2_score(image.ravel(), image_pred.ravel())

        return image_pred, residuos, r2

    def visualizar_ajuste(self, image, image_corrected, residuos):
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
    
    'plano comun'
    def aplanacion(self):
        for key, image in self.img_dict.items():
            if key != 'textura':
                self.img_dict[key]=self.aplanar(image)
                # self.img_dict[key]=self.polinomio_aplanar(image)
                # self.evaluar_ajuste_polinomio(image, self.img_dict[key])

    
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



img_rutas = {'top': 'SENOS1-T.BMP','bottom': 'SENOS1-B.BMP','left': 'SENOS1-L.BMP','right': 'SENOS1-R.BMP','textura': 'SENOS1-S.BMP'}
# img_rutas = {'top': 'CIRC1_T.BMP','bottom': 'CIRC1_B.BMP','left': 'CIRC1_L.BMP','right': 'CIRC1_R.BMP','textura': 'CIRC1.BMP'}

# img_rutas = {'top': 'RUEDA1_T.BMP','bottom': 'RUEDA1_B.BMP','left': 'RUEDA1_L.BMP','right': 'RUEDA1_R.BMP','textura': 'RUEDA1_S.BMP'}
# img_rutas = {'top': 'RUEDA3_T.BMP','bottom': 'RUEDA3_B.BMP','left': 'RUEDA3_L.BMP','right': 'RUEDA3_R.BMP','textura': 'RUEDA3.BMP'}

mi_superficie = reco_superficie3d(img_rutas)

mi_superficie.integracion(c=324, d=34554, z0=0)


mi_superficie.plot_superficie(ver_textura=True)

mi_superficie.contornear_x(20)
    

# mi_superficie.pilacontornos_x(20)


