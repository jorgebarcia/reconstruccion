# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:40:22 2024

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

'''
['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh',
 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8',
 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette',
 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper',
 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white',
 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
'''
# plt.style.use('seaborn-v0_8-ticks')


# plt.style.use('seaborn-v0_8-deep')
# plt.style.use('Solarize_Light2') # Reemplaza 'ggplot' con el nombre del estilo que prefieras
# plt.style.use('seaborn-v0_8-white')
# print(plt.style.available)
# plt.style.use('_mpl-gallery-nogrid')   
# plt.style.use('classic')


def integracion3d(diccionario,c,d,z0,dpixel=1/251.8750,eps=1e-5):
    # notacion de coma flotabte de 32 bits para no perder informacion (llegan en 8 bitss)
    plt.ion()
    i_a=diccionario['right'].astype(np.float32)
    i_b=diccionario['left'].astype(np.float32)
    i_c=diccionario['top'].astype(np.float32)
    i_d=diccionario['bottom'].astype(np.float32)
    
    #restriingimos la division por cero para uqe no nos salte ningun error
    s_dx=(i_a-i_b)/np.clip(i_a+i_b,eps,np.inf)
    s_dy=(i_d-i_c)/np.clip(i_c+i_d,eps,np.inf)
    
    # Acumulación a lo largo de axis=1 --> x / axis=0 -->y
    z_x=cumtrapz(s_dx*c/d, dx=dpixel, axis=1, initial=z0)
    z_y=cumtrapz(s_dy*c/d, dx=dpixel, axis=0, initial=z0)
    z=z_x+z_y
    # z=  gaussian_filter(z, sigma=1)
    
    x,y=np.meshgrid(np.arange(z.shape[1]),np.arange(z.shape[0]))
    # y,x=np.meshgrid(np.arange(z.shape[1]),np.arange(z.shape[0]))

    #es z una funcion que nos lleva de R2 a R1?
    assert z.ndim == 2, "Z no es bidimensional."
    
    'cargamos nuestra textura'   
    textura=cv2.imread(img_ruta['textura'], cv2.IMREAD_COLOR)
    textura=cv2.cvtColor(textura, cv2.COLOR_BGR2RGB)
        
    if textura.shape[0] !=z.shape[0] or textura.shape[1] !=z.shape[1]:
        print(f'La forma de la imagen es: {textura.shape}')
        print(f'La forma de la funcion es: {z.shape}')
        textura=cv2.resize(textura,(z.shape[1],z.shape[0]))
        print('Hemos tenido que reajustar la dimension de la textura por que no coincidia, mira a ver que todo ande bien...')
        
    else: None
    
    'hacemos la figura'
    # plt.style.use('seaborn-v0_8-white')
    reco_3d=plt.figure(figsize=(20,10))
    
    #primera figura
    sin_textura=reco_3d.add_subplot(121,projection='3d')
    sin_textura.plot_surface(x*dpixel , y*dpixel , z, cmap='viridis')
    
    sin_textura.set_title('Topografia sin textura')
    sin_textura.set_xlabel('X (um)')
    sin_textura.set_ylabel('Y (um)')
    sin_textura.set_zlabel('Z (altura en um)')
    
    mappable = cm.ScalarMappable(cmap=cm.viridis)
    mappable.set_array(z)
    plt.colorbar(mappable, ax=sin_textura, orientation='vertical', label='Altura (um)',shrink=0.5,pad=0.2)
    
    #segunda figura
    con_textura=reco_3d.add_subplot(122,projection='3d')
    con_textura.plot_surface(x*dpixel, y*dpixel, z, facecolors=textura/255.0, shade=False)
 
    con_textura.set_title('Topografia con textura')
    con_textura.set_xlabel('X (um)')
    con_textura.set_ylabel('Y (um)')
    con_textura.set_zlabel('Z (altura en um)')
    
    mappable_gray = cm.ScalarMappable(cmap=cm.gray)
    mappable_gray.set_array(textura)  # Asegúrate de que 'textura' está en escala de grises y ajustada.
    plt.colorbar(mappable_gray, ax=con_textura, orientation='vertical', label='Intensidad', shrink=0.5, pad=0.2)
    
    plt.show()
    
    perfil_medio_x = np.mean(z, axis=0)
    ax_x = np.arange(z.shape[1]) * dpixel
    
    # Calcula parámetros de rugosidad para el perfil medio
    Ra_x = np.mean(np.abs(perfil_medio_x - np.mean(perfil_medio_x)))
    Rmax_x = np.max(perfil_medio_x)
    Rmin_x = np.min(perfil_medio_x)
    

    plt.figure(figsize=(15,5))
    plt.plot(ax_x, perfil_medio_x, label='Perfil Medio X')
    
    plt.hlines(Rmax_x, ax_x[0], ax_x[-1], 'r', '--', label='Rmax')
    plt.hlines(Rmin_x, ax_x[0], ax_x[-1], 'b', '--', label='Rmin')
    plt.hlines(np.mean(perfil_medio_x), ax_x[0], ax_x[-1], 'g', '--', label='Media')
    plt.hlines(np.mean(perfil_medio_x) + Ra_x, ax_x[0], ax_x[-1], 'y', '--', label='Ra')
    plt.hlines(np.mean(perfil_medio_x) - Ra_x, ax_x[0], ax_x[-1], 'y', '--')
    
    plt.xlabel('X (μm)')
    plt.ylabel('Altura (μm)')
    plt.title('Perfil Medio de la Superficie a lo Largo del Eje X')
    plt.legend()
    plt.show()
    
    return z

def pilacontornos_x(z,ncontorno,dpixel=1/251.8750):
    pos_y=np.random.randint(0,z.shape[0],ncontorno)
    pos_y=np.sort(pos_y)

    contornos_fig=plt.figure(figsize=(20, 10))
    ax=contornos_fig.add_subplot(111, projection='3d')
    
    contorno_x=np.arange(z.shape[1])*dpixel 
    
    for i in pos_y:
        contorno= z[i,:] #rugosidad concreta
        contorno= gaussian_filter(contorno, sigma=2)
        contorno_y= np.full_like(contorno_x,i*dpixel) 
        ax.plot(contorno_x, contorno_y,contorno, label=f'Perfil en y={i}')        
    ax.set_title('Perfiles de Rugosidad en 3D')
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_zlabel('Z (altura en um)')
    plt.legend()
    plt.show()



def contornear_x(z,pos_y,dpixel=1/251.8750):
    
    pos_y = 20  #20 por ejemplo
    perfil = z[pos_y, :]
    perfil = gaussian_filter(perfil, sigma=1)
    ax_x = np.arange(len(perfil)) * dpixel
    
    'Ra'
    media_perfil=np.mean(perfil)
    
    Ra = np.mean(np.abs(perfil -media_perfil)) #rugosidad media aritmetica  -> promedio abs desviaciones a lo largo de la muestra
    Rmax = np.max(perfil)  
    Rmin = np.min(perfil)
    
    'Rz'
    pikos, _ = find_peaks(perfil, distance=70, prominence=0.2)
    minimos, _ = find_peaks(-perfil, distance=95, prominence=0.2)

    pikos_val=perfil[pikos] #valores nominales pikkos
    minimos_val=perfil[minimos] #valores nominales minimos
    
    #np.argsort(pikos_alturas) --> nos devuelve un array =shape que tiene: [0]- mas bajo....[-1] mas alto
    #nos movemos en el espacio de indices de pikos_val
    pikos_5 = pikos[np.argsort(pikos_val)[-5:]] 
    minimos_5 = minimos[np.argsort(minimos_val)[-5:]]
    
    #pasamos al espacio de indices de perfil a traves del orden hecho
    Rz = np.sum(np.abs(perfil[pikos_5]))/pikos_5.size + np.sum(np.abs(perfil[minimos_5])) / minimos_5.size #por si acaso no hubiese 5 picos
    
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
    
    alto=0.05
    ancho=ax_x[-1]-alto*2
    ax.plot([ancho,ancho], [media_perfil,media_perfil+Ra],'k-',lw=1)
    ax.plot([ancho-alto/2, ancho+alto/2], [media_perfil+Ra, media_perfil+Ra], 'k-', lw=1)
    ax.plot([ancho-alto/2, ancho+alto/2], [media_perfil, media_perfil], 'k-', lw=1)
    
    ax.text(ancho+alto, media_perfil+Ra/2, f'Δz={Ra:.2f}',va='center', ha='left', backgroundcolor='w')
    
    ax.legend()
    plt.show()

def aplanar(imagen):
    '''
    Funcion que elimina las diferencias en las inclinaciones de las imagenes
    recogidas por los detectores BSE
    
    Emplearemos un algoritmo de minimos cuadrados -> deberemos aplanar nuestros arrays par aoperar sobre listas

    Parameters
    ----------
    imagen : imagen fromato .BMP en np.float32 (coma flotante de 32 bits)

    Returns
    -------
    Imagen corregida
    plano corregidor
    '''
    
    index_y,index_x=np.indices(imagen.shape) #obtenemos dos matrices con indices cuya shape=imagen.shape
    imagen_flat=imagen.flatten() #array.dim=1 (valores planos imagen)
    #ahora construimos matriz cada valor plano con sus indices (index_x,index_y,flat_value)
    
    matriz_sist=np.column_stack((index_x.flatten(),index_y.flatten(),np.ones_like(imagen_flat)))  #consume mas meemoria pero es mejor --> np.ones((imagen_flat.shape.. size))
    #z=c1*x+c2*y+c0, c0 es np.ones_like ya que son los valores de intensidad aplanados
    # A es una matriz que cada fila representa un punto x,y + un termino intependiente
    
    'realizamos el ajuste por minimos cuadrados'
    
    coefi,_,_,_=lstsq(matriz_sist,imagen_flat) # _ metodo para desechar variables... solo queremos llstsq[0]--> array.len=3 con coef c1,c2,c0
    # z=c1*x+c2*y+c0
    plano=(coefi[0]*index_x+coefi[1]*index_y+coefi[2]).reshape(imagen.shape)
    
    imagen_correct=imagen-plano
    
    return imagen_correct


def histogrameando(img_dic):

    histo_canva=plt.figure(figsize=(10,20))

    for i, (key,image) in enumerate(img_dic.items()):
        if i<4:                        
            ax_img=histo_canva.add_subplot(4,2,2*i+1)
            imagen=ax_img.imshow(image,cmap='gray')
            ax_img.set_title(f'imagen {key}')
            ax_img.axis('off')
            
            ax_hist=histo_canva.add_subplot(4,2,2*i+2)
            hist=ax_hist.hist(image.ravel(),bins=256,range=[0,256],color='gray',alpha=0.75)
            ax_hist.set_title(f'histograma imagen {key}')
        else: break
    
    # histo_canva.tight_layout()
    plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1,hspace=0.4,wspace=0.3)
    plt.show()
    

img_ruta = {'top': 'SENOS1-T.BMP','bottom': 'SENOS1-B.BMP','left': 'SENOS1-L.BMP','right': 'SENOS1-R.BMP','textura': 'SENOS1-S.BMP'}
# img_ruta = {'top': 'CIRC1_T.BMP','bottom': 'CIRC1_B.BMP','left': 'CIRC1_L.BMP','right': 'CIRC1_R.BMP','textura': 'CIRC1.BMP'}
# img_ruta = {'top': 'RUEDA1_T.BMP','bottom': 'RUEDA1_B.BMP','left': 'RUEDA1_L.BMP','right': 'RUEDA1_R.BMP','textura': 'RUEDA1_S.BMP'}
# img_ruta = {'top': 'RUEDA3_T.BMP','bottom': 'RUEDA3_B.BMP','left': 'RUEDA3_L.BMP','right': 'RUEDA3_R.BMP','textura': 'RUEDA3.BMP'}


if not img_ruta:
    raise ValueError("El diccionario con las fotos esta vacio, espabila!!")



img_dic={key: cv2.imread(image, cv2.IMREAD_GRAYSCALE) for key,image in img_ruta.items()}

for key,image in img_dic.items():
    if image is None:
        raise ValueError(f"La imagen '{key}' no se pudo cargar. Mira a ver que estén bien las rutas.")

# histogrameando(img_dic)

for key, imagen in img_dic.items():
    img_dic[key] = aplanar(imagen)
    
z=integracion3d(img_dic, 1, 1, 0)

contornear_x(z,20)

# pilacontornos_x(z, 3)

# histogrameando(img_dic)

# z=z.flatten()
# with open('mi_archivo.csv', 'a', newline='', encoding='utf-8') as archivo_csv:
#     escritor_csv = csv.writer(archivo_csv)
#     escritor_csv.writerows(z)