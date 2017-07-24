# -*- coding: utf-8 -*-
#################################################
# Alumno: Jorge Ignacio Balanz Pino             #
# Asignatura: Visión Artificial                 #
# Curso: 2016/2017                              #
# Universidad: URJC                             #
#################################################

# REFERENCIAS:
# Reconocer Semillas: https://uvadoc.uva.es/bitstream/10324/17054/3/TFG-P-355.pdf
#

import cv2
import numpy as np
import sys
from transformations.ProcessedContour import ProcessedContour

class ProcessedImage:
    """
    Clase que representa una imagen filtrada y segmentada.
    Mantiene información de:
    Imagen original;
    Imagen transformada y filtrada;
    Objetos (contornos de objetos) detectados en la imagen original;
    Patrones (objetos Pattern) extraídos de la imagen. De cada imagen se pueden
        obtener varios patrones, si la imagen contiene más de un objeto diferenciable
        del fondo: si se localiza más de un contorno en la imagen se tratan por separado.
    """

    AVERAGED_THRESHOLD = -1
    # Umbral por defecto para binarización
    DEFAULT_THRESHOLD = 30
    # Área mínima que debe tener un contorno para ser procesado
    MIN_CONTOUR_AREA = 250
    # Margen que se agrega al contorno para ser tratado individualmente
    CONTOUR_MARGIN = 10
    # Mínima distancia al borde de la imagen para ser tratado
    MIN_BORDER_DISTANCE = 30

    def __init__(self,
                 image: np.ndarray, kernel: int = 3,
                 thold :np.uint8=DEFAULT_THRESHOLD, thold_mode=cv2.THRESH_BINARY,
                 contour_mode=cv2.RETR_EXTERNAL, contour_method=cv2.CHAIN_APPROX_NONE,
                 debug: bool=False
                 ):
        """
        Constructor de la clase.
        Recibe el nombre (con ruta) de un archivo de imagen.
        Carga la imagen y la procesa de tal forma que deja disponible
        la imagen original, la imagen procesada y los contornos localizados en la imagen.
        :param image: Array n-dimensional (numpy.ndarray) con la imagen.
        :param kernel: Tamaño (lado) para elemento estructurante de filtros (Blur y Canny). Por defecto 3
        :param equalize: Indica si se debe ecualizar o no la imagen antes de umbralizar. Por defecto, False
        :param thold: Umbral a utilizar en la umbralización. En caso de no especificarse, se calcula automáticamente.
        :param thold_mode: Modo de umbralización de entre los posibles en CV2. Por defecto, cv2.THRESH_BINARY
        :param contour_mode: Modo de localizar contornos de entre los posibles en CV2. Por defecto, cv2.RETR_EXTERNAL
        :param contour_method: Método para localizar contornos de entre los posibles en CV2. Por defecto, cv2.CHAIN_APPROX_NONE
        """
        """
        Variables de objeto:
            original -> Imagen original en forma de matriz (numpy.ndarray)
            processed -> Imagen procesada y filtrada (numpy.ndarray)
            contours -> Lista de contornos localizados en la imagen.
            descriptors -> Lista de objetos de tipo ProcessedContour que representan los descriptores 
                            y características de un contorno.
        """
        self.original: np.ndarray = None
        self.processed: np.ndarray = None
        self.contours = []
        self.descriptors = []
        if not image is None:
            self.original = image.copy()  # La copio para no afectar lo original
            # Vamos a generar y recoger la imagen generada y los contornos
            self.processImage(kernel, thold, thold_mode, contour_mode, contour_method, debug)
            # Extraemos los descriptores:
            self.__getDescriptors(debug=debug)
            # Presentamos algo de información:
            if debug:
                print("Objetos encontrados y procesados: {}.".format(len(self.descriptors)))
        else:
            raise Exception("\n[ProcessedImage] La imagen de entrada no es válida!\n")

    def processImage(self,
                     kernel:int=3, thold:np.uint8=DEFAULT_THRESHOLD, thold_mode=cv2.THRESH_BINARY,
                     contour_mode=cv2.RETR_EXTERNAL, contour_method=cv2.CHAIN_APPROX_NONE,
                     debug: bool=False
                     ):
        """
        Procesar la imagen que se ha cargado desde el archivo.

        :param kernel: Tamaño (lado) para elemento estructurante de filtros (Blur). Por defecto 3
        :param thold: Umbral a utilizar en la umbralización. En caso de no especificarse, se calcula automáticamente.
        :param thold_mode: Modo de umbralización de entre los posibles en CV2. Por defecto, cv2.THRESH_BINARY
        :param contour_mode: Modo de localizar contornos de entre los posibles en CV2. Por defecto, cv2.RETR_EXTERNAL
        :param contour_method: Método para localizar contornos de entre los posibles en CV2. Por defecto, cv2.CHAIN_APPROX_NONE
        :return: No retorna nada: establece el estado del objeto.
        """
        try:
            # Pasamos la imagen a escala de grises:
            imagen = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)  # Lo recogemos en otra imagen diferente a la original para no afectarla con los cambios
            # Suavizamos (filtro de Media):
            imagen = cv2.blur(imagen, (kernel,kernel))
            # Pasamos a B/N (umbralización):
            # cv2.threshold(<imagen>, <umbral>, <valor al superar umbral>, <tipo de umbralizacion>)
            if thold < 0:
                umbral = cv2.mean(imagen)[0]
                umbral, imagen = cv2.threshold(imagen, umbral, 255, thold_mode)
            elif thold == 0:
                umbral, imagen = cv2.threshold(imagen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thold_mode + cv2.THRESH_OTSU)
            else:
                umbral, imagen = cv2.threshold(imagen, thold, 255, thold_mode)
            # Aplicamos una operación de cierre, para intentar unir las componentes inconexas que
            # posiblemente pertenezcan al mismo objeto:
            imagen = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, (kernel, kernel), iterations=3)
            # Buscamos los contornos dentro de los bordes detectados:
            #http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
            # cv2.findContours(<imagen>, <modo>, <método>)->(imgcontour, contornos, jerarquia)
            # Retorna:
            #   imgcontour - Imagen modificada, donde se han buscado los contornos
            #   contornos - Lista de contornos encontrados
            #   jerarquia - Jerarquía de los contornos encontrados
            (imagen, contornos, jerarquia) = cv2.findContours(imagen, contour_mode, contour_method)
            # Recopilamos la información en el objeto
            rect = []
            rectarea = []
            names = []
            cls = []
            for c in contornos:
                r = cv2.boundingRect(c)
                rect.append(r)
                rectarea.append(r[2]*r[3])
                names.append("")
                cls.append(None)
                if debug:
                    # Dibujamos los contornos, sólo en modo debug
                    cv2.rectangle(self.original, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), (0, 0, 255), 1)
            self.processed = imagen
            self.contours = {"contour": contornos.copy(), "rect": rect.copy(), "area": rectarea.copy(), "name": names.copy(), "class": names.copy()}
            # print(self.contours)
            self.thold = umbral

            # Ya tenemos todos los datos necesarios
            # recogidos en las variables de objeto

            if debug:
                print("Contornos totales encontrados en la imagen: {}.".format(len(contornos)))

        except Exception as e:
            sys.stderr.write("\n[ProcessedImage] Error detectado: {}\n".format(e))
            return None

    def __getDescriptors(self, debug: bool=False):
        """
        Agregar un objeto Pattern, que contienen descriptores,
        a la lista de descriptores, por cada contorno que se haya 
        detectado (si tiene un tamaño mínimo).
        :param debug: Indica si se debe presentar información de debug.
        :return: 
        """
        # Creamos un nuevo objeto Pattern (patrón)
        # ptrn = pat.Pattern
        original = self.original.copy()
        shape = original.shape

        if self.contours != []:
            # Generamos un vector de descriptores por cada contorno
            # si tiene un tamaño mínimo:
            for i in range(len(self.contours["contour"])):
                # Controlamos el tamaño del contorno a tratar
                if self.contours["area"][i] >= ProcessedImage.MIN_CONTOUR_AREA:
                    # Vamos a extraer los descriptores de la subimagen
                    #   (sólo del rectángulo alrededor del contorno):
                    x1 = self.contours["rect"][i][0] - ProcessedImage.CONTOUR_MARGIN
                    x2 = x1 + self.contours["rect"][i][2] + (ProcessedImage.CONTOUR_MARGIN*2)  # Sumamos el ancho
                    y1 = self.contours["rect"][i][1] - ProcessedImage.CONTOUR_MARGIN
                    y2 = y1 + self.contours["rect"][i][3] + (ProcessedImage.CONTOUR_MARGIN*2) # Sumamos el alto
                    # Info del contorno:
                    if debug:
                        print("Procesando contorno {}: posición ({},{}); ancho: {}; alto {};".format(i, x1, y1, self.contours["rect"][i][1], self.contours["rect"][i][3]))
                    # Evitamos lo que se nos haya salido de la imagen: sólo contornos de dentro de la imagen:
                    dif = ProcessedImage.MIN_BORDER_DISTANCE - ProcessedImage.CONTOUR_MARGIN  # Distancia del marco al borde
                    if (x1-dif) > 0 and (x2+dif) < shape[1] and (y1-dif) > 0 and (y2+dif) < shape[0]:
                        # Extraemos la subimagen:
                        img = original[y1:y2, x1:x2, :].copy()

                        # Obtenemos los descriptores del nuevo patrón del contorno
                        ptrn = ProcessedContour(img, rectangle=[x1,y1,x2,y2], debug=debug)

                        # Agregamos el nuevo patrón a la lista de descriptores
                        self.descriptors.append(ptrn)
                    else:
                        if debug:
                            print("  >> El contorno '{}' no se procesa porque está en el borde de la imagen.".format(i))
                else:
                    if debug:
                        print("  >> El contorno '{}' no se procesa porque su área ({}px) es menor de {}px.".format(
                            i, self.contours["area"][i], ProcessedImage.MIN_CONTOUR_AREA))

                        # Fin del FOR (procesado de contornos)
        # Fin del IF de existencia de algún contorno
        else:
            print("  !! [ProcessedImage] No se han localizado contornos en la imagen.")
        if debug:
            self.markAllContours()

    def markAllContours(self, classes: []=[], colors: []=[], windowText: str="Imagen Procesada", show_classification: bool=True):
        """
        Marcar los objetos reconocidos sobre la imagen original, 
        etiquetarlos con la clase que tengan asignada (al aprender o al reconocer) y
        presentarlo en una ventana.
        :param classes: Lista con los nombres de las clases que se pueden reconocer
        :param colors: Lista de colores que se asigna a cada clase (formato (b, g, r) y 
                    lista de igual tamaño que la de nombres) 
        :param windowText: Texto que se presenta en la pantalla.
        :param show_classification: Indica si se debe presentar la imagen generada o sólo se retorna (por defecto, se muestra)
        :return: La imagen con recuadros identificativos de los objetos encontrados.
        """
        img = self.original.copy()
        if classes == [] or colors == [] or not classes or not colors:
            for aux in self.descriptors:
                aux.markContour(img)
        else:
            for aux in self.descriptors:
                # Miramos qué color poner:
                try:
                    x = aux.getCode()
                    i = int(x)
                    aux.markContour(img, colors[i], colors[i])
                except Exception:
                    sys.stderr.write("\n[ProcessedImage] Índice incorrecto (tipo: {}) ... Clasificación: {}.\n".format(type(x), x))
                    aux.markContour(img)
        # Una vez remarcados los objetos, los presento:
        cv2.imshow(windowText, img)
        cv2.waitKey()
        return img

    def default(self):
        return self.__dict__

############  FIN Clase ProcessedImage  ###################################
