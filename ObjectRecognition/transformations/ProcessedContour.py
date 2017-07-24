# -*- coding: utf-8 -*-
#################################################
# Alumno: Jorge Ignacio Balanz Pino             #
# Asignatura: Visión Artificial                 #
# Curso: 2016/2017                              #
# Universidad: URJC                             #
#################################################

import sys
import cv2
import numpy as np
from transformations.Pattern import Pattern

############  CLASE ProcessedContour  #####################################
# Herencia Python: http://python-para-impacientes.blogspot.com.es/2015/06/

class ProcessedContour(Pattern):

    def __init__(self, img: np.ndarray, classification: str="", code=None, rectangle: [int]=[], debug: bool=False):
        Pattern.__init__(self)
        # Creamos atrubutos propios de este tipo de objetos
        self.classification = classification  # Nombre de la clase a la que pertenece (o en la que se clasifica)
        self.code = code  # Código de la clase a la que pertenece (o en la que se clasifica)
        self.rectangle = rectangle
        # Hacemos que el objeto Pattern extraiga los descriptores de la subimagen
        self.extractFromImage(img, debug=debug)

    def getClassification(self):
        if not self.classification:
            return "Sin Clasificar"
        return str(self.classification)

    def getCode(self):
        if self.code is None:
            return -1
        return self.code

    def getRectangle(self):
        if not self.rectangle:
            return ""
        return self.rectangle.copy()

    def markContour(self, img: np.ndarray, rectangle_color: ()=(0, 255, 0), text_color: ()=(255, 255, 255)):
        """
        Recibe una imagen y un contorno procesado e imprime sobre la imagen el rectángulo que envuelve
        contorno y el nombre de clase que tenga el contorno procesado.
        :param img: Imagen sobre la que se imprime
        :param rectangle_color: color del recuadro que se dibujará
        :param text_color: color del texto que se escribirá
        :return: Vacío
        """
        txt = self.getClassification()
        r = self.getRectangle()
        if r:
            cv2.rectangle(img,(r[0], r[1]), (r[2], r[3]), rectangle_color, 2)
            cv2.putText(img, txt, (r[0]+5, r[3]+20), cv2.FONT_HERSHEY_PLAIN, 0.8, color=text_color)
        else:
            sys.stderr.write("\n[ProcesseContour] No hay un rectángulo para el contorno procesado.\n")

    def toString(self):
        txt = "Datos del Contorno Procesado: \n Rectángulo: "
        txt += str(self.rectangle)
        txt += "\n Clasificación actual: "+self.getClassification()
        return txt

############  FIN CLASE ProcessedContour  #####################################