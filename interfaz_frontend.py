import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QGridLayout, QLabel,
                             QWidget, QComboBox, QFileDialog, QGroupBox, QSpacerItem, QSizePolicy, QMessageBox, QDialog,QVBoxLayout,QTabWidget)
from PyQt5.QtGui import QPalette, QColor
# import matplotlib
# matplotlib.use('Qt5Agg')

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from interfaz_soporte import Procesarimagenes, Ecualizacion, Metodosaplanacion, Reconstruccion, Histograma

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import cv2
class Ventana(QMainWindow):
    def __init__(self):
        super().__init__() #obtenemos las propiedades de qmainWindow que nos crea la ventana con cositas

        self.setWindowTitle("Visualizador de Imágenes SEM")
        self.setGeometry(100, 100, 700, 400)  # X pos, Y pos, Width, Height

        #widget central donde se colocara tooodo
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget) #lo establecemos como central

        # hacemos un GridLayout para toda la ventana
        self.mallado_ventana = QGridLayout(self.centralWidget)

        self.img_dict = {'top': None, 'bottom': None, 'left': None, 'right': None
                         }

        #diccionarios de ayuda para la creacion de comparaciones:
        self.img_original={}
        self.img_filtrada = {}

        self.img_original_f={}
        self.img_transfor={}

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
        # self.boton_ruido.setFixedSize(*boton_size)
        self.layout_procesamiento.addWidget(self.boton_ruido,0,0)

        #boton filtro:
        self.boton_filtro = QPushButton("Filtro Gaussiano")
        self.boton_filtro.clicked.connect(self.filtro_gaussiano)
        # self.boton_filtro.setFixedSize(*boton_size)
        self.layout_procesamiento.addWidget(self.boton_filtro,1,0)

        #desplegable filtro:
        self.combox_filtro= QComboBox()
        opciones_filtro=[0.25,0.5, 0.75, 1, 1.5, 2, 3, 4,5]
        for opcion in opciones_filtro:
            self.combox_filtro.addItem(str(opcion),opcion)

        self.layout_procesamiento.addWidget(self.combox_filtro,1,1)

        #boton transformada:

        self.boton_fourier= QPushButton("Transformada de fourier")
        self.boton_fourier.clicked.connect(self.fourier)
        # self.boton_fourier.setFixedSize(*boton_size)
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

        # tabs = QTabWidget()
        #
        # # Crear los widgets que irán en cada pestaña
        # filtro_tab = QWidget()
        # fourier_tab = QWidget()
        #
        # # Agregar widgets a cada pestaña
        # tabs.addTab(filtro_tab, "Filtro")
        # tabs.addTab(fourier_tab, "Fourier")
        #
        # # Crear layouts para cada conjunto de opciones y agregar los widgets correspondientes
        # filtro_layout = QVBoxLayout()
        # # ... Agregar opciones de filtro al filtro_layout ...
        # filtro_tab.setLayout(filtro_layout)
        #
        # fourier_layout = QVBoxLayout()
        # # ... Agregar opciones de fourier al fourier_layout ...
        # fourier_tab.setLayout(fourier_layout)
        #
        # # Agregar el widget de pestañas al layout principal
        # self.mallado_ventana.addWidget(tabs, 0, 2)


        self.caja_resultados = QGroupBox("Resultados del Procesamiento")
        self.layout_resultados = QGridLayout(self.caja_resultados)

        # luego actualizamos el texto
        self.label_resultados = QLabel("Aquí se mostrán los resultados : \n")
        self.layout_resultados.addWidget(self.label_resultados, 0, 0)

        self.mallado_ventana.addWidget(self.caja_resultados,0, 2,3,1) #posicion caja_resultados -- > se extiende por 3 filas y 1 col

        # Desplegable comparacion filtro
        self.combox_image = QComboBox()
        self.combox_image.addItem("---FILTRO---Seleccionar una imagen para ver comparacion---")
        opciones_imagen = ['top', 'bottom', 'left', 'right']
        self.combox_image.currentIndexChanged.connect(self.ver_filtro)
        for opcion in opciones_imagen:
            self.combox_image.addItem(opcion)

        self.layout_resultados.addWidget(self.combox_image, 1, 0)

        self.label_img_original = QLabel()
        self.label_img_filtrada = QLabel()

        self.label_img_original.setMinimumSize(200, 200)
        self.label_img_filtrada.setMinimumSize(200, 200)

        self.layout_resultados.addWidget(self.label_img_original, 2, 0)
        self.layout_resultados.addWidget(self.label_img_filtrada, 2, 1)

        #desplegable fourier:
        self.combox_fourier =QComboBox()
        self.combox_fourier.addItem("---FOURIER---Seleccionar una imagen para ver compacion---")
        self.combox_fourier.currentIndexChanged.connect(self.ver_fourier)
        for opcion in opciones_imagen:
            self.combox_fourier.addItem(opcion)
        self.layout_resultados.addWidget(self.combox_fourier,5,0)
        self.label_img_original_f=QLabel()
        self.label_img_trans=QLabel()
        self.label_img_original_f.setMinimumSize(200, 200)
        self.label_img_trans.setMinimumSize(200, 200)
        self.layout_resultados.addWidget(self.label_img_original_f, 6, 0)
        self.layout_resultados.addWidget(self.label_img_trans, 6, 1)



        'CAJA HISTOGRAMA'

        self.caja_histograma=QGroupBox("Histograma")
        self.layout_histograma =  QGridLayout(self.caja_histograma)

        self.label_histograma = QLabel("Histograma en tiempo real")
        self.layout_histograma.addWidget(self.label_histograma, 0, 0)

        self.mallado_ventana.addWidget(self.caja_histograma,1,0)

        #boton hist
        self.boton_histograma = QPushButton("Ver Histograma")
        self.boton_histograma.clicked.connect(self.histograma)
        self.layout_histograma.addWidget(self.boton_histograma,1, 0)

        'CAJA INTEGRACION Y PLOT'
        self.caja_reco = QGroupBox("reconstruccion")
        self.layout_reco = QGridLayout(self.caja_reco)

        self.label_reco= QLabel("Pulsar para integrar "
                                "\n y ver")
        self.layout_reco.addWidget(self.label_reco,2,0)

        self.mallado_ventana.addWidget(self.caja_reco,1,1)

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

    def cv_a_qp(self,image):
        # print("1")
        h, w = image.shape[:2]
        # print("2")
        if len(image.shape) == 3: #pue estara e rgbb
            image_qt=QImage(image.data, w,h,3*w, QImage.Format_RGB888)
            print('ns pq pero tas en rgb')
        else:
            # print('3')
            image_qt = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
            # print('4')
            image_qt = image_qt.scaled(self.label_img_original.width(), self.label_img_original.height(),
                                        Qt.KeepAspectRatio)
            # print("5")
        return QPixmap.fromImage(image_qt)

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
        try:
            sigma=self.combox_filtro.currentData()
            procesador = Procesarimagenes(self.img_dict)
            self.img_original,self.img_filtrada=procesador.filtro(sigma=float(sigma))
            text_result= "Se han filtrado correctament las imagenes"
            self.label_resultados.setText(text_result)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ocurrió un error durante la aplicación del filtro: {e}")

    def ver_filtro(self): #NO ACTUA EN BUCLE!! ACTUA SOBRE EL CURRENT TEXT!!
        # ref=self.combox_image.currentText().lower()
        ref = self.combox_image.currentText()
        if ref == "---FILTRO---Seleccionar una imagen para ver comparacion---":
            self.label_img_original.clear()
            self.label_img_filtrada.clear()
            self.label_txt_original.clear()
            self.label_txt_filtrada.clear()
            return
        if ref.lower() not in self.img_original:
            QMessageBox.warning(self, "Advertencia","No hay ningun filtro aplicado")
            # print('1')
            self.combox_image.currentIndexChanged.disconnect()
            self.combox_image.setCurrentIndex(0)
            self.combox_image.currentIndexChanged.connect(self.ver_filtro)
            # print('2')
            return
        try:
            original_np_img = self.img_original[ref]
            filtered_np_img = self.img_filtrada[ref]

            original_pixmap = self.cv_a_qp(original_np_img)
            filtered_pixmap = self.cv_a_qp(filtered_np_img)

            self.label_img_original.setPixmap(original_pixmap)
            self.label_img_filtrada.setPixmap(filtered_pixmap)

            self.label_txt_original = QLabel("Pre - Filtrado")
            self.label_txt_filtrada = QLabel("Post - Filtrado")

            self.label_txt_original.setAlignment(Qt.AlignHCenter)
            self.label_txt_filtrada.setAlignment(Qt.AlignHCenter)

            self.layout_resultados.addWidget(self.label_txt_original, 3, 0)
            self.layout_resultados.addWidget(self.label_txt_filtrada, 3, 1)
            # self.label_img_original.repaint()
            # self.label_img_filtrada.repaint()
            # print(self.img_original[ref])
            # print(self.img_original[ref].shape)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ocurrió un error durante el plot de comparacion: {e}")

    def fourier(self):
        if not self.verificar_carga():
            QMessageBox.warning(self,"OYE","TIENES QUE CARGAR TODAS LAS IMAGENES PRIMEROOO")
        else:
            procesador = Procesarimagenes(self.img_dict)
            ssim_dict,psnr_dict,r,self.img_original_f,self.img_transfor=procesador.aplicar_fourier()
            text_result = "Resultados de Fourier:\n"
            for key in self.img_dict.keys():
                text_result += (f"Imagen {key} - "
                                f"SSIM: {ssim_dict[key]:.4f}, "
                                f"PSNR: {psnr_dict[key]:.4f}, "
                                f"Radio: {r}\n")
            self.label_resultados.setText(text_result)
    def ver_fourier(self):
        ref = self.combox_fourier.currentText()
        if ref == "---FOURIER---Seleccionar una imagen para ver comparacion---":
            self.label_img_original_f.clear()
            self.label_img_trans.clear()
            self.label_txt_original_f.clear()
            self.label_txt_trans.clear()
            return
        if ref.lower() not in self.img_original_f:
            QMessageBox.warning(self, "ADvertencia", "No hay ninguna transformada aplicada")
            self.combox_fourier.currentIndexChanged.disconnect()
            self.combox_fourier.setCurrentIndex(0)
            self.combox_fourier.currentIndexChanged.connect(self.ver_fourier)
            return
        try:
            print("original")
            # print("1")
            original_f_np = self.img_original_f[ref]
            # print("1.1")
            trans_np = self.img_transfor[ref]
            # print("2")
            original_f_pixmap = self.cv_a_qp(original_f_np)
            trans_pixmap = self.cv_a_qp(trans_np)
            # print("3")
            self.label_img_original_f.setPixmap(original_f_pixmap)
            self.label_img_trans.setPixmap(trans_pixmap)
            # print("4")
            # self.label_img_original.setPixmap(self.img_original_f[image])
            # self.label_img_filtrada.setPixmap(self.img_dict[image])
            # print("5")
            self.label_txt_original_f = QLabel("Pre - Transformada")
            self.label_txt_trans = QLabel("Post - Trasformada")
            # print("6")
            self.label_txt_original_f.setAlignment(Qt.AlignHCenter)
            self.label_txt_trans.setAlignment(Qt.AlignHCenter)
            # print("7")
            self.layout_resultados.addWidget(self.label_txt_original_f, 7, 0)
            self.layout_resultados.addWidget(self.label_txt_trans, 7, 1)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ocurrió un error durante el plot de comparacion: {e}")

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

    def plotear(self):
        if not self.verificar_carga():
            QMessageBox.warning(self, "OYE", "TIENES QUE CARGAR TODAS LAS IMAGENES PRIMEROOO")
        else:
            try:
                reconstructor=Reconstruccion(self.img_dict,self.textura if hasattr(self, 'textura') else None)
                reconstructor.integracion(1,1,0)

                self.ventana_sin=QDialog(self)
                layout_sin=QVBoxLayout(self.ventana_sin)
                canvas_sin=FigureCanvas(Figure(figsize=(10,8)))
                layout_sin.addWidget(canvas_sin)

                reconstructor.plot_superficie(canvas_sin.figure,ver_textura=False)
                self.ventana_sin.setWindowTitle("Superficie Reconstruida")
                self.ventana_sin.show()
                # print(self.textura)

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
    # app =  QtWidgets.QApplication(sys.argv)

    # estilo_global = """
    # QWidget {
    #     font-size: 14px;
    #     color: #33333;
    #     background-color: #FFFFFF;
    # }
    #
    # QPushButton {
    #     background-color: #F0F0F0;
    #     border: 1px solid #BBBBBB;
    #     padding: 5px;
    #     border-radius: 4px;
    #     min-height: 20px;
    # }
    #
    # QPushButton:hover {
    #     background-color: #D5D5D5;
    # }
    #
    # QPushButton:pressed {
    #     background-color: #CCCCCC;
    # }
    #
    # QGroupBox {
    #     border: 1px solid #BBBBBB;
    #     border-radius: 4px;
    #     margin-top: 20px;
    # }
    #
    # QGroupBox::title {
    #     subcontrol-origin: margin;
    #     subcontrol-position: top center;
    #     padding-left: 10px;
    #     padding-right: 10px;
    #     padding-top: 10px;
    # }
    #
    # QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {
    #     border: 1px solid #BBBBBB;
    #     border-radius: 4px;
    #     padding: 2px;
    #     background-color: #EFEFEF;
    #     min-height: 20px;
    # }
    #
    # QLabel, QCheckBox {
    #     margin: 2px;
    # }
    #
    # QSlider::groove:horizontal {
    #     border: 1px solid #BBBBBB;
    #     height: 8px;
    #     background: #F0F0F0;
    #     margin: 2px 0;
    # }
    #
    # QSlider::handle:horizontal {
    #     background: #D5D5D5;
    #     border: 1px solid #BBBBBB;
    #     width: 18px;
    #     margin: -2px 0;
    #     border-radius: 4px;
    # }
    #
    # QSlider::add-page:horizontal {
    #     background: #BBBBBB;
    # }
    #
    # QSlider::sub-page:horizontal {
    #     background: #D5D5D5;
    # }
    # """
    # app.setStyleSheet(estilo_global)
    app.setStyle('Fusion')
    # app.setStyle('Windows')

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(15, 15, 15))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(palette)

    mainWin = Ventana()
    mainWin.show()
    sys.exit(app.exec_())

app = QApplication([])
print(QApplication.style().metaObject().className())
print(QApplication.availableStyles())

