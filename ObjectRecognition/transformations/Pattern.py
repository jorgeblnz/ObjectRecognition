# -*- coding: utf-8 -*-
#################################################
# Alumno: Jorge Ignacio Balanz Pino             #
# Asignatura: Visión Artificial                 #
# Curso: 2016/2017                              #
# Universidad: URJC                             #
#                                               #
# NOTA: aunque hay funciones de esta clase que  #
#   no se utilizan en esta práctica, se dejan   #
#   para futuras consultas del código.          #
#################################################

import sys
import cv2
import numpy as np

################  CLASE Pattern  ##########################################

class Pattern():
    """
    Clase que representa un patrón (un vector de descriptores).
    Recopila descriptores de 5 tipos:
    1.- De forma (SHAPE_Features; FEATURES_SHAPE): redondez
    2.- De color (COLOR_Features; FEATURES_COLOR): media_azul, media_verde, media_roja, histograma_RGB, histograma_HSV
    3.- De Hu (HuMoments; FEATURES_HUMOMENTS): Momentos de Hu (7 momentos de Hu)
    4.- SURF - Speeded-Up Robust Features -(SURF_Features; FEATURES_SURF):  
        Un vector con un descriptor por cada punto característico detectado;
        Cada descriptor, a su vez = vector de 64 elementos
    5.- SIFT - Scale Invariant Feature Transform -(SURF_Features; FEATURES_SURF):  
        Un vector con un descriptor por cada punto característico detectado;
        Cada descriptor, a su vez = vector de 128 elementos

    Los patrones se pueden obtener accediendo directamente a las variables de objeto o usando el método
    getFeatures, que acepta como argumento los valores FEATURES_XXX definidos en la clase. 
        Para conseguir varios tipos de descriptores agregados en un solo vector, usar la operación binaria OR (|)
        Los descriptores se agregan en el orden que se ha descrito en el listado de tipos.

    La mejor combinación que he encontrado ha sido:
    Unir descriptores de forma con descriptores de color;
    Normalizar los descriptores de color para que tengan un peso similar al de forma;
    Ajustar el área de la forma haciendo una media ponderada entre el área del contorno 
        y el área del rectángulo que lo envuelve, con la fórmula: 
        area = (area_rectang + (3 * area_contorno)) / 4
    Se ha conseguido un porcentaje de acierto del 93% (aprox.) con las imágenes facilitadas de aprendizaje y test.
    """

    FEATURES_SHAPE = 1
    FEATURES_COLOR = 2
    FEATURES_SURF = 4
    FEATURES_HUMOMENTS = 8
    FEATURES_SIFT = 16
    DEFAULT_FEATURES = FEATURES_COLOR | FEATURES_SHAPE

    SHAPE_WEIGHT = 1.0


    AVERAGED_THRESHOLD = -1
    DEFAULT_THRESHOLD = 30
    MIN_CONTOUR_PERCENT_AREA = 30

    def __init__(self):
        self.SHAPE_Features = []
        self.COLOR_Features = []
        self.SURF_Features = []
        self.__SURF_points = None
        self.SIFT_Features = []
        self.__SIFT_points = None
        self.HuMoments = []
        self.__is_loaded = False

    def getFeatures(self, features: int = DEFAULT_FEATURES, debug: bool=False):
        x = []
        if not self.__is_loaded:
            if debug:
                print("\n  !! [Pattern] No se pueden retornar descriptores porque no han sido cargados.")
            return None
        if features < 1 or features > 32:
            raise IndexError(
                "[Pattern] No se pueden retornar descriptores con la denominación solicitada (index<{}>).".format(
                    features))
        if features & Pattern.FEATURES_SHAPE: # Únicamente la redondez (1)
            if debug:
                print("Se añaden de forma: {}".format(self.SHAPE_Features))
            if self.SHAPE_Features == []:
                self.SHAPE_Features.append(0.0)
            x.extend(self.SHAPE_Features)
        if features & Pattern.FEATURES_COLOR: # Medias de colores (3) + hist-color (1) + hist-HSV (1)
            if debug:
                print("Se añaden de color: {}".format(self.COLOR_Features))
            x.extend(self.COLOR_Features)
        if features & Pattern.FEATURES_HUMOMENTS: # Momentos Hu (7)
            if debug:
                print("Se añaden Momentos Hu: {}".format(self.HuMoments))
            x.extend(self.HuMoments)
        if features & Pattern.FEATURES_SURF:
            if debug:
                print("Se añaden SURF: {}".format(self.SURF_Features))
            x.extend(self.SURF_Features)
        if features & Pattern.FEATURES_SIFT:
            if debug:
                print("Se añaden SIFT: {}".format(self.SIFT_Features))
            x.extend(self.SIFT_Features)
        if debug:
            print("Final: {}".format(x))
        return x

    # Descriptores de color (método operativo): ofrecen mejores resultados (máx. 93.24%), al menos en el caso de esta práctica.
    # Más lento que el método que se deja como alternativo (__getColorDescriptors2)
    def __getColorDescriptors(self, image: np.ndarray = None, debug: bool = False) -> []:
        if image is None:
            raise Exception("[Pattern] No se ha recibido una imagen para extraer descriptores de color.")
        # Probamos a ecualizar la imagen para realzar la imagen y aumentar el contraste:
        # self.__equalize(image, mode="YUV")  # Aceptable: 85.91549295774648%
        # self.__equalize(image)  # Aceptable: 85.91549295774648%
        # SIN ecualizar: 85.91549295774648%  => La ecualización no aporta mejoras => No se aplica
        z = image.reshape((-1, 3))
        z = np.float32(z)
        # Referencias de K-Means
        # http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
        # http://reyesalfonso.blogspot.com.es/2010/08/implementacion-de-k-means-en-opencv.html
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
        # Segmentar una imagen por color, clasificando los pixeles que tienen colores parecidos en un mismo grupo:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        # Con el algoritmo de K-Medias se calculan 3 centros sobre los que agrupar colores
        compactness, labels, centers = cv2.kmeans(z, 3, None, criteria, 10, flags)
        # En centers tenemos los centros calculados con las K-Medias
        # Cada center es una color BGR => Calculamos la media (para poder ordenar por "intensidad media")
        means = []
        #  Los valores se normalizan dividiendo entre el valor máximo (componentes de color de 8bits -> 255)
        # Así se iguala su peso con el resto de descriptores:
        centers = (centers/255)
        for centro in centers:
            means.append(sum(centro) / len(centro))
        # De los 3 valores obtenidos nos quedamos con los 2 mayores,
        #  que representan los colores más significativos.
        #  Se usarán las "coordenadas 3D" de esos centros más significativos (componentes BGR).
        if max(means) == means[0]:
            if min(means) == means[2]:
                # Centros 0 y 1
                self.COLOR_Features.extend(centers[0])
                self.COLOR_Features.extend(centers[1])
            else:
                # Centros 0 y 2
                self.COLOR_Features.extend(centers[0])
                self.COLOR_Features.extend(centers[2])
        elif max(means) == means[1]:
            if min(means) == means[2]:
                # Centros 0 y 1
                self.COLOR_Features.extend(centers[1])
                self.COLOR_Features.extend(centers[0])
            else:
                # Centros 1 y 2
                self.COLOR_Features.extend(centers[1])
                self.COLOR_Features.extend(centers[2])
        else:
            if min(means) == means[0]:
                # Centros 2 y 1
                self.COLOR_Features.extend(centers[2])
                self.COLOR_Features.extend(centers[1])
            else:
                # Centros 0 y 2
                self.COLOR_Features.extend(centers[2])
                self.COLOR_Features.extend(centers[0])
        if debug:
            print(means)
        return self.COLOR_Features.copy()

    # Descriptores de color alternativos: ofrecen peores resultados (máx. 87.84%), al menos en el caso de esta práctica.
    # Más rápido que el método que se deja operativo (__getColorDescriptors)
    # Conservo el método para posibles futuras consultas.
    def __getColorDescriptors2(self, image: np.ndarray = None,
                              add_hist_hsv: bool=False, add_hist_bgr: bool=True,
                              debug: bool = True) -> []:
        # Referencias sobre colores
        # http://arantxa.ii.uam.es/~jms/pfcsteleco/lecturas/20110318OscarBoullosa.pdf
        # http://hpclab.ucentral.edu.co/wiki/index.php/Descriptores_de_color
        #
        # http://www.pyimagesearch.com/2014/03/03/charizard-explains-describe-quantify-image-using-feature-vectors/
        # https://robologs.net/2014/07/02/deteccion-de-colores-con-opencv-y-python/

        # Primeros descriptores: las medias de los canales de color
        # mean retorna lista con la media de los colores BGR
        if image is None:
            raise Exception("[Pattern] No se ha recibido una imagen para extraer descriptores de color.")

        #img = self.__equalize(image, 'YUV')  # MAL: 67.56% // Sin Histogramas: 78.37837837837837%%  // Sólo H-HSV: 74.32432432432432% // Sólo H-RGB: 50%
        #img = self.__equalize(image)  # MUY MAL: 37%     // Sin Histogramas: 50% // Sólo H-HSV: 48%  // Sólo H-RGB: 41%
        img = image.copy()  # Sin EQ, Aceptable: máx. 87.83783783783784% sin histogramas o con el histograma BGR sólo; 79% con ambos o con el HSV sólo;
        # Conclusión de los resultados: Los histogramas, en este caso, no aportan nada positivo!!
        # Conclusión de los resultados: Las ecualizaciones, en este caso, no aportan nada positivo!!

        rep = 0.0
        intens = 32
        desc = cv2.mean(img)  # Recoge las medias de los canales de color (+ otro canal: alpha -transparencia-)
        desc = desc[0:3]  # Quitamos el último canal que no aporta información
        desc = [x / 255 for x in desc]  # Valores Normalizados (se presuponen canales de 8 bits)

        # Siguientes descriptores con el histograma "aplanado", para todos los canales
        # Hacemos distinción de 'intens' intensidades de color para coger rangos de color y
        #   además facilitar cómputo.

        # Histograma BGR
        # Calculamos los histogramas de los canales:
        if add_hist_bgr:
            h: np.ndarray
            if img.ndim > 2:
                if img.shape[2] > 1:
                    h = cv2.calcHist([img], [0], None, [intens], [0, 255])
                else:
                    h = cv2.calcHist([img], [0, 1, 2], None, [intens, intens, intens], [0, 255, 0, 255, 0, 255])
            else:
                raise Exception(
                    "[Pattern] La imagen para extraer descriptores de color sólo tiene {} dimensiones.".format(img.ndim))
            # Aplanamos el histograma para sacar otro descriptor:
            h = h.flatten()
            # Agregamos al vector el color más representativo de la imagen:
            rep = np.argmax(h)
            if rep.dtype == 'int32':
                rep2 = float(rep) / float(intens)  # Normalizado según el num de rangos de color
            else:
                sys.stderr.write("\n[Pattern] Error al extraer descriptor del histograma RGB (tipo<{}>; dims<{}>).\n".format(rep.dtype, rep.shape))
                rep2 = 0.0
            desc.append(rep2)

        # Histograma HSV
        if add_hist_hsv:
            imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Calculamos los histogramas de los canales:
            if img.shape[2] > 1:
                h = cv2.calcHist([imghsv], [0], None, [intens], [0, 255])
            else:
                h = cv2.calcHist([imghsv], [0, 1, 2], None, [intens, intens, intens], [0, 255, 0, 255, 0, 255])
            # Aplanamos el histograma para sacar otros descriptores:
            h = h.flatten()
            # Agregamos al vector el rango más representativo de la imagen HSV:
            rep = np.argmax(h)
            if rep.dtype == 'int32':
                rep2 = float(rep) / float(intens)  # Normalizado según el num de rangos de color
            else:
                sys.stderr.write("\n[Pattern] Error al extraer descriptor del histograma HSV (tipo<{}>; dims<{}>).\n".format(rep.dtype, rep.shape))
                rep2 = 0.0
            desc.append(rep2)

        if debug:
            print("Descriptores de color: {}".format(desc))
            cv2.imshow("Ventana para Color (RGB)", img)
            cv2.imshow("Ventana para Color (HSV)", imghsv)
            cv2.waitKey()

        self.COLOR_Features = desc
        return self.COLOR_Features.copy()

    def __equalize(self, img: np.ndarray, mode: str='RGB', debug: bool=False)->np.ndarray:
        """
        Ecualización del histograma de una imagen en color
        :param img: imagen a ecualizar
        :param mode: modo "YUV" (ecualiza histograma desde componentes YUV) o 
                    modo "RGB" (ecualiza hitograma de componentes RGB)
        :param debug: Indica si se muestra información adicional de depuración.
        :return: La imagen ecualizada.
        """
        img_eq = img.copy()
        # Vamos a ecualizar la imagen para mejorar los contrastes:
        # La ecualización se aplica a un canal, así que hace falta un "truco" para equalizar colores:
        if mode == 'YUV':
            # Convertimos la imagen a formato YUV (Y - Iluminancia(B/N); U y V - Color)
            # Referencias (enhance contrast color image opencv):
            # http://shubhamagrawal.com/opencv/opencv-playing-with-brightness-contrast-histogram-blurness/
            # http://cromwell-intl.com/3d/histogram/
            img_eq = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            if debug:
                print(img_eq)
                cv2.imshow("Pasado a YUV", img_eq)
            # Ecualizamos el canal Y (iluminancia - B/N)
            if not img_eq is None:
                img_eq[:, :, 0] = cv2.equalizeHist(img_eq[:, :, 0])
                if debug:
                    cv2.imshow("Imagen YUV ecualizada (canal Y)", img_eq)
                # Lo volvemos a pasar a RGB
                img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)
        else:
            # Opción 2: Ecualizar los canales RGB, de uno en uno:
            img_eq[:, :, 0] = cv2.equalizeHist(img_eq[:, :, 0])
            img_eq[:, :, 1] = cv2.equalizeHist(img_eq[:, :, 1])
            img_eq[:, :, 2] = cv2.equalizeHist(img_eq[:, :, 2])

        if debug:
            cv2.imshow("Ecualizado en {}".format(mode), img_eq)
            cv2.waitKey()
        return img_eq

    def __getContourDescriptors(self, contour, rect: [], debug: bool = False):
        """
        Recibe como parámetro un contorno para extraer de él los descriptores correspondientes.
        Extrae los descriptores de forma (redondez) y los Momentos de Hu.
        El resto de parámetros son opcionales: son ajustes más detallados para extraer los contornos.
        """
        # Recogemos datos sobre el contorno obtenido:
        # perimeter = (2 * rect[2]) + (2 * rect[2])  -> Peores resultados!!!
        perimeter = cv2.arcLength(contour, True)
        # Calculamos el área con respecto al rectángulo que envuelve al contorno:
        # Las sombras de las imágenes pueden hacer los contornos irregulares.
        # => Área será el ancho por alto:
        area = rect[2]*rect[3]  # Máximo obtenido área de rectángulo: 87.83783783783784%
        # Intentamos ajustarno más al área real del contorno:
        carea = cv2.contourArea(contour)
        # area = (area + carea) / 2 # Máximo con media: 87.83783783783784%
        # area = (area + 2 * carea) / 3  # Máximo con media ponderada: 90.54054054054053%
        area = (area + 3 * carea) / 4  # Máximo con media ponderada: 93.24324324324324%
        # area = (area + 4 * carea) / 5  # Máximo con media ponderada: 93.24324324324324%
        # area = (area + 5 * carea) / 6  # Máximo con media ponderada: ~91%
        if area > 0:  # Me aseguro que no sea 0, aunque viene ya comprobado...
            # Factor redondez (ejemplo): http://revistas.unal.edu.co/index.php/acta_agronomica/article/view/52212/56904
            # La redondez será un valor entre 0 y 1 (normalizado)
            roundness = (perimeter ** 2) / (4 * np.pi * area)
        # Agregamos la redondez a descriptores de forma
        # Mejor resultado obtenido: área ponderada [area = (area + 3 * carea) / 4] con peso = 1  => No aplico peso
        # roundness = roundness * Pattern.SHAPE_WEIGHT
        self.SHAPE_Features.append(roundness)
        if debug:
            print("Parámetros del contorno: Área={}; Perímetro={}; \"Redondez\"={}".format(area, perimeter, roundness))

        # cv2.moments(<contorno>) -> Calcula los momentos a partir de un contorno
        moments = cv2.moments(contour)
        if debug:
            print("Momentos del contorno: {}".format(moments))
        # cv2.HuMoments(<momentos>) -> Calcula los momentos invariantes de Hu a partir de los momentos (calculados con cv2.moments)
        huMoments = cv2.HuMoments(moments)
        # HuMoments sólo -> 47.30%
        # HuMoments + color -> 87.84%
        # HuMoments + color + shape -> 89.19%

        if debug:
            print("Momentos Hu del contorno: {}".format(huMoments))
        # Agregamos los momentos como descriptores
        for x in huMoments:
            self.HuMoments.append(x[0])
        # Retornamos lo obtenido:
        r = self.SHAPE_Features.copy()
        r.extend(self.HuMoments)
        return r

    # http://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
    # http://arantxa.ii.uam.es/~jms/pfcsteleco/lecturas/20110318OscarBoullosa.pdf
    # SIFT (Scale Invariant Feature Transform) y SURF (Speeded Up Robust Features)
    # (SIFT) -pág 44- fue desarrollado por Lowe [8] como un algoritmo capaz de detectar puntos característicos estables en una imagen.
    #  Basado en DoG: Difference-of-Gaussians basado en el espacio-escala Gaussiano de una imagen L(x; y; Desv);
    #   El espacio-escala Gaussiano es definido como la convolución de funciones 2D Gaussianas
    #    G(x; y; Desv) con diferentes valores de Desv sobre la imagen original.
    #   Cada punto de interés = vector de 128 componentes.
    #   Octava: conjunto de las imágenes Gaussianas suavizadas junto con las imágenes DoG.
    #   https://ianlondon.github.io/blog/how-to-sift-opencv/

    def __getSIFTDescriptors(self, img: np.ndarray, debug: bool = False):
        """
        Obtener descriptores mediante algoritmo SIFT
        :param img: Imagen de la que se extraen los descriptores
        :param debug: Indica si se debe mostrar información de depuración.
        :return: Una lista de descriptores (cada descriptor = vector de 128 elementos)
        """
        # Primero se convierte la imagen a escala de grises:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Obtenemos un objeto SIFT:
        sift = cv2.xfeatures2d.SIFT_create()
        # Ahora obtenemos los puntos característicos y los descriptores (cada descriptor es un vector de 128 elementos)
        self.__SIFT_points, self.SIFT_Features = sift.detectAndCompute(gray, None)
        if debug:
            cv2.imshow("Puntos SIFT", cv2.drawKeypoints(gray, self.__SIFT_points, img.copy()))
            # print(self.SIFT_Features)
            print("---> Puntos SIFT generados: {}".format(len(self.SIFT_Features)))
        return self.SIFT_Features.copy()

    # SURF es como SIFT, pero parece que es un algoritmo mejorado.
    # Se basa en matriz Hessiana para los cálculos.
    def __getSURFDescriptors(self, img: np.ndarray, debug: bool = False):
        """
        Obtener descriptores mediante algoritmo SIFT
        :param img: Imagen de la que se extraen los descriptores
        :param debug: Indica si se debe mostrar información de depuración.
        :return: Una lista de descriptores (cada descriptor = vector de 64 elementos)
        """
        # Primero se convierte la imagen a escala de grises:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Obtenemos un objeto SIFT:
        sift = cv2.xfeatures2d.SURF_create()
        # Ahora obtenemos los puntos característicos y los descriptores (cada descriptor es un vector de 64 elementos)
        self.__SURF_points, self.SURF_Features = sift.detectAndCompute(gray, None)
        if debug:
            cv2.imshow("Puntos SURF", cv2.drawKeypoints(gray, self.__SURF_points, img.copy()))
            print(self.SURF_Features)
            print("---> Puntos SURF generados: {}".format(len(self.SURF_Features)))
        return self.SURF_Features.copy()

    def extractFromImage(self, image: np.ndarray, kernel: int = 3,
                         thold: np.uint8=DEFAULT_THRESHOLD, thold_mode=cv2.THRESH_BINARY,
                         contour_mode=cv2.RETR_EXTERNAL, contour_method=cv2.CHAIN_APPROX_NONE,
                         debug: bool = False
                         ):
        try:
            # Pasamos la imagen a escala de grises:
            imagen = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Lo recogemos en otra imagen
            if debug:
                cv2.imshow("Imagen en Grises", imagen)
            # Suavizamos (filtro de Media):
            imagen = cv2.blur(imagen, (kernel, kernel))
            if debug:
                cv2.imshow("Imagen en Suavizada", imagen)
            # Pasamos a B/N (umbralización):
            # cv2.threshold(<imagen>, <umbral>, <valor al superar umbral>, <tipo de umbralizacion>)
            if thold < 0:
                umbral = cv2.mean(imagen)[0]
                umbral, imagen = cv2.threshold(imagen, umbral, 255, thold_mode)
            elif thold == 0:
                umbral, imagen = cv2.threshold(imagen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               thold_mode + cv2.THRESH_OTSU)
            else:
                umbral, imagen = cv2.threshold(imagen, thold, 255, thold_mode)
            # Aplicamos una operación de cierre, para intentar unir las componentes inconexas que
            # posiblemente pertenezcan al mismo objeto:
            imagen = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, (kernel, kernel), iterations=3)
            if debug:
                cv2.imshow("Imagen en B/N (umbral {})".format(umbral), imagen)
            # Buscamos los contornos dentro de los bordes detectados:
            # http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
            # cv2.findContours(<imagen>, <modo>, <método>)->(imgcontour, contornos, jerarquia)
            # Retorna:
            #   imgcontour - Imagen modificada, donde se han buscado los contornos
            #   contornos - Lista de contornos encontrados
            #   jerarquia - Jerarquía de los contornos encontrados
            (imagen, contornos, jerarquia) = cv2.findContours(imagen, contour_mode, contour_method)
            # YA TENEMOS LOS CONTORNOS!

            # Recopilamos la información en el objeto
            descriptors = []
            shape = image.shape
            total_area = 0.0
            total_area = float(shape[0]) * shape[1]
            max_area = 0.0
            max_index = 0
            max_rect = []
            for i in range(len(contornos)):
                # Se supone que la imagen que llega aquí sólo tiene un contorno que procesar...
                # Buscamos el más grande.
                # El área lo cogemos del rectángulo envolvente:
                #  el contorno detectado puede estar "deforme" (sonbras y huecos)
                r = cv2.boundingRect(contornos[i])
                area = r[2]*r[3]  # Multiplicamos ancho*alto (en ese orden)
                if area > max_area:
                    max_area = area
                    max_index = i
                    max_rect = r

            # Comprobamos que tenga un tamaño mínimo (en porcentaje)
            if (max_area / total_area) > (Pattern.MIN_CONTOUR_PERCENT_AREA / 100):
                # Ahora que estamos seguros de que el contorno cumple las condiciones,
                #   realizamos la extracción de descriptores:

                # Extraemos descriptores de color (de la imagen de entrada, directamente)
                self.__getColorDescriptors(image, debug=debug)
                # Extraemos descriptores del contorno: de forma y los invariantes de Hu
                self.__getContourDescriptors(contornos[max_index], max_rect, debug=debug)
                # TODO: Habilitar el resto de descriptores sólo si es necesario
                # Extraemos los descriptores SURF
                # self.__getSURFDescriptors(image, debug)
                # Extraemos los descriptores SIFT
                # self.__getSIFTDescriptors(image, debug)

                # Marco el objeto como cargado
                self.__is_loaded = True

            # Fin del if contorno > porcentaje
            else:
                print("\n  !! [Pattern] No se encuentra un contorno con tamaño mínimo en la imagen tratada.")
            if debug:
                print("Contornos encontrados: {}...".format(len(contornos)))
                cv2.waitKey()
            # Retornamos los descriptores extraídos en este método
            f = self.SHAPE_Features.copy()
            f.extend(self.HuMoments)
            return f

        except Exception as e:
            sys.stderr.write("\n[Pattern] Error al extraer descriptores del contorno:\n {}\n".format(e))
            return None

    def default(self):
        return self.__dict__

################  FIN CLASE Pattern  ##########################################
