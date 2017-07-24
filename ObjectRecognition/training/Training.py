# -*- coding: utf-8 -*-
#################################################
# Alumno: Jorge Ignacio Balanz Pino             #
# Asignatura: Visión Artificial                 #
# Curso: 2016/2017                              #
# Universidad: URJC                             #
#################################################

import sys
import os
import time
import cv2
import transformations.ProcessedImage as ip
from recognition.Recognizer import Recognizer

################  CLASE Training  #####################################

class Training:
    """
    Entrenamiento para ajustar el clasificador.
    A partir de un directorio base, crea una clase por cada subdirectorio y extrae
     los patrones de entrenamiento de esa clase, de las imágenes contenidas 
     en el subdirectorio.
    """
    AUTO_TRAINING = 0
    MANUAL_TRAINING = 1

    def __init__(self, classname: str, training_path: str, training_mode: int=MANUAL_TRAINING):
        self.name = classname
        self.__mode = training_mode
        self.__training_path = os.path.join(os.getcwd(), training_path)
        self.__classnames = []
        self.__classcodes = []


    def learning(self, debug: bool=False)->(Recognizer, [], []):
        """
        Realizar el aprendizaje a partir de unos ficheros de entrada.
        Se recibe la ruta a un directorio y cada uno de sus subdirectorios se considera 
        un conjunto de imágenes de entrenamiento para una clase diferente.
        :param source: Directorio donde se encuentran las imágenes para el aprendizaje.
        :param destination: Fichero de destino para los datos aprendidos.
        :return: El array con los parámetros de los descriptores que se han aprendido.
        """
        # En este método sólo se hace el control de errores
        try:
            if self.__training_path is None or not os.path.isdir(self.__training_path):
                raise IOError("No se encuentra el directorio '{}'...".format(self.__training_path))

            # Lanzamos el entrenamiento
            return self.__letsTrain()

        except IOError as ioe:
            sys.stderr.write("\n[Training] I/O Error: {}.\n".format(ioe))
            return None, [], []
        except OSError as ose:
            sys.stderr.write("\n[Training] OS Error: {}.\n".format(ose))
            return None, [], []
        except Exception as e:
            sys.stderr.write("[Training] Exception: {0}.".format(e))
            return None, [], []

    def __letsTrain(self, debug: bool=False)->(Recognizer, [], []):
        # Crear un objeto Clasificador
        # Recorrer todos los subdirectorios:
        # (Referencias de recorrido de directorios y archivos:
        #   http://www.sromero.org/wiki/programacion/tutoriales/python/recorrer_arbol
        #   https://docs.python.org/3/library/os.path.html
        #   https://www.tutorialspoint.com/python/os_walk.htm
        # En cada subdirectorio
        #   -> Procesar todas las imágenes del subdirectorio.
        #   Por cada imagen:
        #       -> Crear un patrón (o más, si los hay)
        #       -> Meterlo en el clasificador como patrón de entrenamiento con el nombre de la clase correspondiente al directorio.
        # Finalmente, guardar el clasificador

        # Creamos el clasificador:
        clsfr = Recognizer()
        codes = []  # Listado de códigos de clase (Núm. de subrirectorio: se añade uno por cada patrón procesado)
        samples = []  # Listado de patrones de entrenamiento (se añade uno por cada patrón procesado)

        # Recorremos los subdirectorios:
        for base, directories, files in os.walk(self.__training_path):
            # directories contiene el nombre de los subdirectorios
            # Ahora vamos a crear una clase (clase de objetos a clasificar) para cada subdirectorio:
            c = 0  # Número/Código de subdirectorio
            for dirs in directories:
                # Recojo el contador de directorio como código identificador de la clase asociada al directorio
                self.__classcodes.append(c)
                # Reseteo el contador de ficheros del directorio:
                filecount = 1
                for base2, subdir, subfiles in os.walk(os.path.join(self.__training_path, dirs)):
                    print("\n--------------------------------------------------------------------------------")
                    print(">>> {} <<< Procesando Directorio '{}':".format(c, base2))
                    print("    Ficheros a procesar: {}".format(len(subfiles)))
                    proc_img: ip.ProcessedImage = None

                    clsname = ""
                    # Si es en modo manual, se pide un nombre de clase para cada subdirectorio;
                    # Si el modo es automático se asignan nombres automáticamente:
                    if self.__mode == Training.AUTO_TRAINING:
                        clsname = dirs+"-[Cod. "+str(c)+"]" # Nombre de directorio (for externo) + contador de directorio
                    else:  # Por defecto, modo manual
                        clsname = self.__getNewClassName(subfiles, base2)

                    # Si tenemos un nombre para la clase, procesamos el directorio:
                    if clsname:
                        self.__classnames.append(clsname)
                        # Ahora vamos a crear patrones por cada imagen de muestra y recogerlo en los objetos correspondientes
                        num_of_samples = 0
                        filecount = 0
                        for imagefile in subfiles:
                            (fname, fext) = os.path.splitext(imagefile)
                            if fext.lower() == ".jpg" or fext.lower() == ".png" or fext.lower() == ".bmp":
                                f = os.path.join(base2, imagefile)
                                # print("({0}.-{1})-> Procesando imagen {2} ....".format(dirs, filecount+1, f))
                                if debug:
                                    print("\n({0}.{1})-> Procesando imagen {2} .... \n".format(dirs, filecount+1, f))
                                img = cv2.imread(f)
                                # Filtramos y segmentamos: extraemos contornos de la imagen (pueden ser varios)
                                #  y sacamos los descriptores en forma de objetos "Pattern"
                                proc_img = ip.ProcessedImage(img)
                                for pattern in proc_img.descriptors:
                                    # Por si hiciese falta, incluyo el nombre de la clase al objeto
                                    pattern.classification = clsname
                                    pattern.code = c
                                    # Lo añadimos a los listados de datos de entrenamiento
                                    sample = pattern.getFeatures()
                                    if sample:
                                        samples.append(sample) # Agregamos un nuevo patrón de entrenamiento
                                        codes.append(pattern.code)  # El nombre de la clase igual en el directorio entero
                                        num_of_samples += 1  # Incrementamos contador
                                    else:
                                        print("  !! [Training] Archivo '{}':\n\t\t\tSe ha localizado un objeto del que no se ha podido extraer un patrón... (Valor: {})".format(f, sample))
                                filecount += 1
                            else:
                                print("  !! [Training] El fichero '{}' no es de imagen (sólo png, jpg y bmp).".format(imagefile))
                    else:
                        print("  !! [Training] El subdirectorio '{}' no se ha procesado: no se ha introducido un nombre de clase válido.".format(subdir))

                    # Incrementamos el contador de directorios procesados
                    c += 1
                    # Presentamos una pequeña estadística:
                    print("\n# Se han procesado {} archivos del subdirectorio '{}'.".format(filecount, base2))
                    # Mostramos número de muestras tratadas:
                    print("# Clase asignada '{0}': {1} patrones de entrenamiento procesados.\n".format(clsname, num_of_samples))
                    # Cuando hemos procesado todas las imágenes del subdirectorio, agregamos la clase al clasificador:
                    if not proc_img is None:
                        # Agregamos el nombre de la clase
                        clsfr.classnames.append(clsname)
                    else:
                        sys.stderr.write("\n[Training] No se han obtenido patrones de entrenamiento del directorio '{}'.\n".format(dirs))

        # Ya ha terminado la extracción de patrones!
        # Cargamos todos los patrones de entrenamiento de todos los subdirectorios en el clasificador:
        # Cada vez que se entrena, elimina los entrenamientos previos => todos a la vez.
        if samples and codes:
            clsfr.train(samples, codes)
        # Presentamos información:
        print("\n-------------------------------------------------------------------------------")
        print("-->  El clasificador KNN ha sido entrenado con {} patrones de entrenamiento.".format(len(samples)))
        print("-->  Se han creado un total de {} clases: \n {}".format(len(clsfr.classnames), clsfr.classnames))
        print("-------------------------------------------------------------------------------\n")

        # Retornamos el clasificador, y los nombres y códigos de las clases
        return clsfr, self.__classnames.copy(), self.__classcodes.copy()

    def __getNewClassName(self, subfiles: [], base):
        extension = ""
        clsname = None
        i = 0
        while extension == "" and i < len(subfiles):
            (fname, fext) = os.path.splitext(subfiles[i])
            if fext.lower() == ".jpg" or fext.lower() == ".png" or fext.lower() == ".bmp":
                extension = fext
            else:
                i += 1
        if i >= len(subfiles):
            sys.stderr.write("\nNo hay archivos de  imagen (bmp, jpg, png) en el directorio {}\n".format(base))
            return None
        else:
            print("\nFichero Seleccionado como muestra: '{}'...".format(subfiles[i], base))
            img = cv2.imread(os.path.join(base, subfiles[i]))
            winname = "Imagen de Muestra..."
            cv2.namedWindow(winname)
            cv2.imshow(winname, img)
            print("Indique nombre para la clase de la imagen de muestra (subdir. '{0}'):".format(base))
            cv2.waitKey()
            clsname = input()
            cv2.destroyWindow(winname)
        return clsname

    def default(self):
        return self.__dict__


################  FIN CLASE Training  #####################################
