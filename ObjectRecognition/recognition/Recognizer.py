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
#   SÓLO SE IMPLEMENTA CLASIFICADOR KNN !!      #
#################################################

#
# Los clasificadores "ml" derivan de StatModel
# http://docs.opencv.org/3.1.0/db/d7d/classcv_1_1ml_1_1StatModel.html
# Esta clase, a su vez, deriva de Algorithm
# http://docs.opencv.org/3.1.0/d3/d46/classcv_1_1Algorithm.html
#
# Referencias KNN:
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html
# Existe otro llamado FLANN:
# Fast Library for Approximate Nearest Neighbors (FLANN)
# Flann se utiliza para SIFT y SURF

import cv2
import numpy as np
# ATENCIÓN: ml asume que todos los patrones son del mismo tamaño
import cv2.ml as ml

class Recognizer:
    """
    Clase que representa un clasificador.
    Se puede entrenar y, después de este proceso, es capaz de reconocer
    objetos similares (o iguales) a los aprendidos en el entrenamiento. 
    """

    KNN_CLASSIFIER = 1
    FLANN_CLASSIFIER = 2
    BAYES_CLASSIFIER = 4
    MLP_CLASSIFIER = 8


    def __init__(self):

        # Clasificador FLANN (implementación de KNN): Se usa para SIFT y SURF
        # http://docs.opencv.org/3.1.0/dc/de2/classcv_1_1FlannBasedMatcher.html
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        # Es subclase de DescriptorMatcher
        # http://docs.opencv.org/3.1.0/db/d39/classcv_1_1DescriptorMatcher.html
        # Parámetros para el clasificador FLANN
        #index_params = dict(algorithm=ml.KNEAREST_KDTREE, trees=5)
        #search_params = dict(checks=50)  # or pass empty dictionary
        # Crear un objeto de este tipo:
        # self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Clasificador Bayesiano:
        # http://docs.opencv.org/3.1.0/d4/d8e/classcv_1_1ml_1_1NormalBayesClassifier.html
        #self.__bayes = ml.NormalBayesClassifier_create()

        # Multi_Layer Perceptron:
        # http://docs.opencv.org/3.1.0/d0/dce/classcv_1_1ml_1_1ANN__MLP.html
        #self.__mlp = ml.ANN_MLP_create()

        # Datos de Entrenamiento:
        # http://docs.opencv.org/3.1.0/dc/d32/classcv_1_1ml_1_1TrainData.html
        # self.trainPatterns = ml.TrainData_create()

        # K-Nearest Neighbours:
        # http://docs.opencv.org/3.2.0/d5/d26/tutorial_py_knn_understanding.html
        #
        self.knn = cv2.ml.KNearest_create()

        # Índices de los nombres de las clases (lo que retorna y con lo que se entrena el clasificador):
        self.classindexes = None # La declaro, pero no la relleno: se define en el entrenamiento
        # Nombres de las clases que se clasifican:
        self.classnames = []  # La declaro, pero no la relleno: se define en el entrenamiento
        # Tamaño de los patrones con que se trabaja:
        self.patternsize = -1

    def train(self, train_data: [], class_codes: [], type: int=KNN_CLASSIFIER, debug: bool=False):
        """
        Entrenar un clasificador con datos de entrenamiento
        :param train_codes: Array de tipo float. Cada fila es un patrón de entrenamiento y cada columna un descriptor del patrón.
        :param class_names: Array de tipo numérico con los identificadores de las clases de los patrones.
        :param type:
        :return: Número de patrones de aprendizaje incorporados
        """
        if type < 1 or type > 16:
            raise IndexError(
                "[Recognizer] No se puede entrenar un clasificador con la denominación solicitada (index<{}>).".format(type))
        if not class_codes or class_codes == []:
            raise IndexError(
                "[Recognizer] No se han introducido códigos de clases válidos ({}).".format(class_codes))
        if not train_data or train_data == []:
            raise IndexError(
                "[Recognizer] No se han introducido datos de entrenamiento válidos.")

        # Cojo el tamaño de los patrones:
        self.patternsize = len(train_data[0])

        # Debemos crear arrays numpy de valores float que es lo que admiten las funciones de entrenamiento:
        # Una que representa los índices de los nombres de las clases (para poder retornarlo, después, en forma de texto):
        self.classindexes = np.array(class_codes, dtype=np.float32)  # Lo convertimos a formato adecuado: np.array de float32
        self.classindexes = self.classindexes.reshape((self.classindexes.size, 1))  #Aseguramos que tiene las dimensiones adecuadas
        data = np.float32(train_data)

        # Información de debug:
        if debug:
            print("Vamos a entrenar el clasificador con:")
            for i in range(len(data)):
                print("{} .- {} --> Clase {}; {}".format(i + 1, data[i], self.classindexes[i],
                                                         self.classnames[int(self.classindexes[i])]))
        # Ahora vamos con el entrenamiento:
        if type & Recognizer.KNN_CLASSIFIER:
            self.knn.train(data, cv2.ml.ROW_SAMPLE, self.classindexes)
        else:
            raise Exception("[Recognizer] La clase Recognizer sólo implementa el clasificador KNN, actualmente.")

        return len(train_data)

    def recognize(self, newpattern: [], type: int=KNN_CLASSIFIER, debug: bool=False):
        if type < 1 or type > 16:
            raise IndexError(
                "[Recognizer] No se puede utilizar un clasificador con la denominación solicitada (index<{}>).".format(type))
        if not newpattern or newpattern == []:
            raise IndexError(
                "[Recognizer] No se ha introducido un patrón válido.")

        # Convertimos la entrada a formato adecuado:
        data = np.float32(newpattern)
        data = data.reshape((1, len(data)))

        # Ahora, lo clasificamos
        if type & Recognizer.KNN_CLASSIFIER:
            if not self.knn.isTrained():
                raise Exception(
                    "[Recognizer] El clasificador KNN aún no ha sido entrenado.")
            # findNearest(<patrón>, <K-vecinos>)
            # Busca el mayor número de coincidencias entre los K vecinos más cercanos (de entre los patrones cargados)
            #  Si k=1 => busca el más cercano
            # retorna:
            #   ret = El resultado final según el algoritmo KNN
            #   results = Etiquetas de los k vecinos más cercamos
            #   neighbours = Los k vecinos más cercanos (patrones)
            #   dist = Las correspondientes distancias a los vecinos más cercanos.
            ret, results, neighbours, dist = self.knn.findNearest(data, k=3)
            # Resultados: k=1 -> 91.89%; k=2 -> 91.89%%; k=3 -> 93.24%; k=4 -> 93.24%; k=5 -> 90.54%
            if debug:
                print("@@ Clasificación con KNN = {} (ret); Results: {}\nNeighbours: {}\nDistances: {}".format(
                    ret, results, neighbours,dist))
            # return ret, results, neighbours, dist
        else:
            raise Exception("[Recognizer] La clase Recognizer sólo implementa el clasificador KNN, actualmente.")

        if type & Recognizer.MLP_CLASSIFIER:
            if not self.__mlp.isTrained():
                raise Exception(
                    "[Recognizer] El clasificador MLP aún no ha sido entrenado.")
            else:
                raise Exception("[Recognizer] El clasificador MLP aún no ha sido implementado.")
            ret, results = self.__mlp.predict([data])
            if debug:
                print("@@ Clasificación con MLP = {} (ret); Results: {}\n".format(ret, results))

        return ret, results

    def classify(self, newpattern: [], type: int=KNN_CLASSIFIER, debug: bool=False)->(str, int):
        """
        Clasificar un patrón con el clasificador que se indique
        :param newpattern: Patrón que se debe clasificar
        :param type: Tipo de clasificador que se debe emplea (actualmente sólo KNN_CLASSIFIER; por defecto)
        :param debug: Indica si se debe presentar información adicional de depuración.
        :return: Un string que representa la clase en que se clasifica el patrón evaluado y un entero que representa el código de esa clase.
        """
        ret, results = self.recognize(newpattern, type, debug)
        aux = int(ret)
        return self.classnames[aux], aux
