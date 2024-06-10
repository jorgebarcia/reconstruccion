import cv2
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve


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
class Reconstruccion:
    def __init__(self, datos):
        self.datos = datos
        self.calculo_gradientes(c=85.36, d=100, eps=1e-5, ver=True)
        self.z = self.solve_lyapunov_equation()
        self.plot_superficie(ver_textura=True)

    def calculo_gradientes(self, c, d, eps=1e-5, ver=True):
        i_a = self.datos['right'].astype(np.float32)
        i_b = self.datos['left'].astype(np.float32)
        i_c = self.datos['top'].astype(np.float32)
        i_d = self.datos['bottom'].astype(np.float32)

        factor = c / d
        self.s_dx = factor * (i_a - i_b) / np.clip(i_a + i_b, eps, np.inf)
        self.s_dy = factor * (i_d - i_c) / np.clip(i_c + i_d, eps, np.inf)

        if ver:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title('Gradient X')
            plt.imshow(self.s_dx, cmap='jet', interpolation='nearest')
            plt.colorbar()
            plt.figure()
            plt.title('Gradient Y')
            plt.imshow(self.s_dy, cmap='jet', interpolation='nearest')
            plt.colorbar()
            plt.show()

    def solve_lyapunov_equation(self):
        n, m = self.s_dx.shape

        # Crear operadores de diferenciación
        Dx = diags([-1, 1], [0, 1], shape=(n - 1, n)).toarray()
        Dy = diags([-1, 1], [0, 1], shape=(m - 1, m)).toarray()

        # Construir matrices laplacianas
        Lx = Dx.T @ Dx
        Ly = Dy.T @ Dy

        # Crear la matriz Laplaciana combinada
        Laplacian = kron(eye(m), Lx) + kron(Ly, eye(n))

        # Resolver el sistema usando un solucionador para matrices dispersas
        # Crear la matriz rhs b
        div_sx = np.zeros_like(self.s_dx)
        div_sx[:-1, :] = self.s_dx[1:, :] - self.s_dx[:-1, :]
        div_sx = div_sx.flatten()

        div_sy = np.zeros_like(self.s_dy)
        div_sy[:, :-1] = self.s_dy[:, 1:] - self.s_dy[:, :-1]
        div_sy = div_sy.flatten()

        b = div_sx + div_sy

        # Resolver la ecuación usando el método de gradiente conjugado
        Z_flat = spsolve(Laplacian, b)
        Z = Z_flat.reshape(n, m)

        return Z

    def plot_superficie(self, ver_textura=True):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(self.z.shape[0]), np.arange(self.z.shape[1]))
        ax.plot_surface(X, Y, self.z.T, cmap='viridis')
        plt.show()


# Ejemplo de uso:
img_rutas = {'top': 'calibrado/0_4-T.BMP', 'bottom': 'calibrado/0_4-B.BMP', 'left': 'calibrado/0_4-L.BMP',
             'right': 'calibrado/0_4-R.BMP', 'textura': 'calibrado/0_4-S.BMP'} #son 960 x 1280 pixeles

cargar = Cargarimagenes(img_rutas)

reconstruir = Reconstruccion(cargar)

