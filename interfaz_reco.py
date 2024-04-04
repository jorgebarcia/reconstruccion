import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QGridLayout, QLabel,
                             QWidget, QFileDialog, QGroupBox, QSpacerItem, QSizePolicy, QMessageBox, QDialog,QVBoxLayout)

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from soporte_interfaz import Procesarimagenes, Ecualizacion, Metodosaplanacion, Reconstruccion, Histograma

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import cv2
class Ventana(QMainWindow):
    def __init__(self):
        super().__init__() #obtenemos las propiedades de qmainWindow que nos crea la ventana con cositas

        self.setWindowTitle("Visualizador de Imágenes SEM")
        self.setGeometry(100, 100, 1400, 700)  # X pos, Y pos, Width, Height

        #widget central donde se colocara tooodo
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget) #lo establecemos como central

        # hacemos un GridLayout para toda la ventana
        self.mallado_ventana = QGridLayout(self.centralWidget)

        self.img_dict = {'top': None, 'bottom': None, 'left': None, 'right': None
                         }

        "facemos una QGroupBox (cajita) pa los botones de carga"
        '------------------------------------------------------'

        self.caja_upload=QGroupBox("Imagenes")
        self.grid_layout = QGridLayout(self.caja_upload)

        #botones caja
        boton_size = (100, 30)
        # Top
        self.boton_top=QPushButton("Imagen Top")
        self.boton_top.setFixedSize(*boton_size)
        self.boton_top.clicked.connect(lambda:self.upload_images('top'))
        self.grid_layout.addWidget(self.boton_top,0,0)
        #bottom
        self.boton_bottom = QPushButton("Imagen Bottom")
        self.boton_bottom.setFixedSize(*boton_size)
        self.boton_bottom.clicked.connect(lambda: self.upload_images('bottom'))
        self.grid_layout.addWidget(self.boton_bottom,1,0)
        #left
        self.boton_left = QPushButton("Imagen left")
        self.boton_left.setFixedSize(*boton_size)
        self.boton_left.clicked.connect(lambda: self.upload_images('left'))
        self.grid_layout.addWidget(self.boton_left,2,0)
        #right
        self.boton_right = QPushButton("Imagen right")
        self.boton_right.setFixedSize(*boton_size)
        self.boton_right.clicked.connect(lambda: self.upload_images('right'))
        self.grid_layout.addWidget(self.boton_right,3,0)
        #textura
        self.boton_textura = QPushButton("Imagen Textura")
        self.boton_textura.setFixedSize(*boton_size)
        self.boton_textura.clicked.connect(lambda: self.upload_images('textura'))
        self.grid_layout.addWidget(self.boton_textura,4,0)

        # self.caja_upload.setMaximumWidth(200)
        # self.caja_upload
        #ahora que esta creado, añadimos el Qgrupbox al layout principal de la ventana
        self.mallado_ventana.addWidget(self.caja_upload,0,0)

        #apretamos la caja a la izq
        spacer_right = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.mallado_ventana.addItem(spacer_right, 0, 2)
        spacer_bottom = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.mallado_ventana.addItem(spacer_bottom, 1, 0, 1, 2)

        # Qlabel pa ver que se cargan bien
        self.label_status = QLabel("Debes cargar las"
                                   "imagenes")
        self.grid_layout.addWidget(self.label_status, 5, 0)  # Fila 5, Columna 0

        # Hacer que el layout ocupe solo una parte de la ventana
        self.grid_layout.setColumnStretch(1, 0)  # Hacer que la columna 1 tenga un peso mayor


        'CAJA PROCESAMIENTO'
        '--------------------------------------------------------------------------------'
        self.caja_procesamiento = QGroupBox("Opciones de Procesamiento")
        self.layout_procesamiento = QGridLayout(self.caja_procesamiento)

        #boton ruido
        self.boton_ruido = QPushButton("Nivel Ruido")
        self.boton_ruido.clicked.connect(self.ruido)
        self.layout_procesamiento.addWidget(self.boton_ruido,0,0)

        #boton filtro:
        self.boton_filtro = QPushButton("Filtro Gaussiano")
        self.boton_filtro.clicked.connect(self.filtro_gaussiano)
        self.layout_procesamiento.addWidget(self.boton_filtro,1,0)
        #boton transformada:

        self.boton_fourier= QPushButton("Transformada de fourier")
        self.boton_fourier.clicked.connect(self.fourier)
        self.layout_procesamiento.addWidget(self.boton_fourier,2,0)

        # botton ecualizar
        self.boton_ecualizar= QPushButton("Ecualizar")
        self.boton_ecualizar.clicked.connect(self.ecualizando)
        self.layout_procesamiento.addWidget(self.boton_ecualizar,3,0)

        # boton aplanar
        self.boton_aplanar= QPushButton("Aplanar")
        self.boton_aplanar.clicked.connect(self.aplanar)
        self.layout_procesamiento.addWidget(self.boton_aplanar,4,0)



        #añadimos esta caja al layout principal de la ventana

        self.mallado_ventana.addWidget(self.caja_procesamiento,0,1)

        'CAJA RESULTADOS DE PROCESAMIENTO'
        '-----------------------------------'

        self.caja_resultados = QGroupBox("Resultados del Procesamiento")
        self.layout_resultados = QGridLayout(self.caja_resultados)

        # luego actualizamos el texto
        self.label_resultados = QLabel("Niveles de ruido: \n")
        self.layout_resultados.addWidget(self.label_resultados, 0, 0)

        self.mallado_ventana.addWidget(self.caja_resultados, 0, 2)


        'CAJA HISTOGRAMA'

        self.caja_histograma=QGroupBox("Histograma")
        self.layout_histograma =  QGridLayout(self.caja_histograma)

        self.label_histograma = QLabel("Histograma en tiempo real")
        self.layout_histograma.addWidget(self.label_histograma, 0, 0)

        self.mallado_ventana.addWidget(self.caja_histograma,1,0)

        #boton hist
        self.boton_histograma = QPushButton("Ver Histograma")
        self.boton_histograma.clicked.connect(self.histograma)
        self.layout_histograma.addWidget(self.boton_histograma, 0, 0)

        'CAJA INTEGRACION Y PLOT'
        self.caja_reco = QGroupBox("reconstruccion")
        self.layout_reco = QGridLayout(self.caja_reco)

        self.label_reco= QLabel("Pulsar para integrar "
                                "\n y ver")
        self.layout_reco.addWidget(self.label_reco,2,0)

        self.mallado_ventana.addWidget(self.caja_reco,1,1)

        #boton integrar
        # self.boton_intg= QPushButton("Integrar")
        # self.boton_intg.clicked.connect(self.integrar)
        # self.layout_reco.addWidget(self.boton_intg,0,0)

        #boton plotear:
        self.boton_plotear=QPushButton("plotear")
        self.boton_plotear.clicked.connect(self.plotear)
        self.layout_reco.addWidget(self.boton_plotear,1,0)

    def upload_images(self, image_type):
        try:
            filePath, _ = QFileDialog.getOpenFileName(self, f"Cargar imagen {image_type}", "", "Images (*.bmp)")
            if filePath:
                # self.img_dict[image_type] = QPixmap(filePath) #solo si queremos verlas al cargar (NO)
                if image_type != 'textura':
                    self.img_dict[image_type] = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
                else:
                    self.textura = cv2.imread(filePath, cv2.IMREAD_COLOR)
                self.label_status.setText(f"La imagen {image_type} se ha cargado correctamente")
            else:
                self.label_status.setText(f"No subiste la foto")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            print(e)

    def verificar_carga(self):
        #necesitamos que ninguna imagen tenga un valor none para poder ejecutar cualquier linea
        return all(value is not None for value in self.img_dict.values()) #vaya protip

    def ruido(self):
        if not self.verificar_carga():
            QMessageBox.warning(self,"OYE","TIENES QUE CARGAR TODAS LAS IMAGENES PRIMEROOO")
        else:
            # print('1')
            # print(self.img_dict)
            procesador = Procesarimagenes(self.img_dict)
            # print('2')
            result=procesador.nivel_ruido()
            # print('3')
            text_result= "Niveles de ruido:\n" + "\n".join(f"{k}: {v}" for k, v in result.items())
            self.label_resultados.setText(text_result)

    def filtro_gaussiano(self):
        if not self.verificar_carga():
            QMessageBox.warning(self,"OYE","TIENES QUE CARGAR TODAS LAS IMAGENES PRIMEROOO")
        else:
            procesador = Procesarimagenes(self.img_dict)
            result=procesador.filtro(ver=False)
            text_result= "Se han filtrado correctament las imagenes"
            self.label_resultados.setText(text_result)

    def fourier(self):
        if not self.verificar_carga():
            QMessageBox.warning(self,"OYE","TIENES QUE CARGAR TODAS LAS IMAGENES PRIMEROOO")
        else:
            procesador = Procesarimagenes(self.img_dict)
            ssim_dict,psnr_dict,r=procesador.aplicar_fourier(ver=False)
            text_result = "Resultados de Fourier:\n"
            for key in self.img_dict.keys():
                text_result += (f"Imagen {key} - "
                                f"SSIM: {ssim_dict[key]:.4f}, "
                                f"PSNR: {psnr_dict[key]:.4f}, "
                                f"Radio: {r}\n")
            self.label_resultados.setText(text_result)

    def ecualizando(self):
        if not self.verificar_carga():
            QMessageBox.warning(self,"OYE","TIENES QUE CARGAR TODAS LAS IMAGENES PRIMEROOO")
        else:
            ecualizador= Ecualizacion(self.img_dict)
            contraste_dict, entropia_dict =ecualizador.ecualizar()
            text_result = "Resultados de Ecualizacion:\n"
            for key in self.img_dict.keys():
                text_result += (f"Imagen {key} - "
                                f"Mejora de Contraste: {contraste_dict[key]:.4f}, "
                                f"Mejora de Entropia: {entropia_dict[key]:.4f})\n"
                                )
            self.label_resultados.setText(text_result)


    def aplanar(self):
        if not self.verificar_carga():
            QMessageBox.warning(self,"OYE","TIENES QUE CARGAR TODAS LAS IMAGENES PRIMEROOO")
        else:
            aplanar=Metodosaplanacion(img_dict=self.img_dict)
            condicion_dict, residuo_dict =aplanar.aplicar_aplanacion()
            text_result = "Resultados de Aplanar por minimos cuadrados:\n"
            for key in self.img_dict.keys():
                text_result += (f"Imagen {key} - "
                                f"Numero de condicion: {condicion_dict[key]:.4f}, "
                                f"Valor del Residuo: {residuo_dict[key]:.4f})\n"
                                )
            self.label_resultados.setText(text_result)

    def histograma(self):
        if not self.verificar_carga():
            QMessageBox.warning(self, "Advertencia", "Necesitas cargar todas las imágenes primero.")
        else:
            try:
                self.ventana_histograma=QDialog(self)
                self.ventana_histograma.setWindowTitle("Histograma de las imágenes")
                layout_histograma = QVBoxLayout(self.ventana_histograma)

                canvas_histograma= FigureCanvas(Figure(figsize=(10,20)))
                # añadimos el canva, al layout
                layout_histograma.addWidget(canvas_histograma)

                histograma = Histograma(self.img_dict)
                histograma.histogramear(canvas_histograma.figure)
                #printamos por pantalla
                self.ventana_histograma.exec_()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Ocurrió un error al mostrar el histograma: {e}")

    # def integrar(self):
    #     if not self.verificar_carga():
    #         QMessageBox.warning(self,"OYE","TIENES QUE CARGAR TODAS LAS IMAGENES PRIMEROOO")
    #     else:
    #         try:
    #             integrar=Reconstruccion(img_dict=self.img_dict)
    #             integrar.integracion(1,1,1)
    #             text_result="Se ha reconstruido correctamente"
    #             self.label_reco.setText(text_result)
    #         except Exception as e:
    #             QMessageBox.critical(self, "Error", f"Ocurrió un error al integrar: {e}")

    def plotear(self):
        if not self.verificar_carga():
            QMessageBox.warning(self, "OYE", "TIENES QUE CARGAR TODAS LAS IMAGENES PRIMEROOO")
        else:
            try:
                reconstructor=Reconstruccion(self.img_dict)
                reconstructor.integracion(1,1,0)

                self.ventana_sin=QDialog(self)
                layout_sin=QVBoxLayout(self.ventana_sin)
                canvas_sin=FigureCanvas(Figure(figsize=(10,8)))
                layout_sin.addWidget(canvas_sin)

                # reconstructor=Reconstruccion(img_dict=self)
                reconstructor.plot_superficie(canvas_sin.figure,ver_textura=False)
                self.ventana_sin.setWindowTitle("Superficie Reconstruida")
                self.ventana_sin.show()

                if hasattr(self,'textura') and self.textura is not None:
                    self.ventana_con = QDialog(self)
                    layout_con = QVBoxLayout(self.ventana_con)
                    canvas_con = FigureCanvas(Figure(figsize=(10, 8)))
                    layout_con.addWidget(canvas_con)

                    reconstructor.plot_superficie(canvas_con.figure, ver_textura=True)
                    self.ventana_con.setWindowTitle("Superficie Reconstruida con textura")
                    self.ventana_con.show()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Ocurrió un error al plotear en 3D: {e}")







if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = Ventana()
    mainWin.show()
    sys.exit(app.exec_())


# class GestionarFunciones:
#     def __init__(self,img_dict):
#         self.img_dict=img_dict
#
#     def ruido(self):
#         procesador = Procesarimagenes(self.img_dict)
#         return procesador.nivel_ruido()
