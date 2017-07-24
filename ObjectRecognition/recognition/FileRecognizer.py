# -*- coding: utf-8 -*-
#################################################
# Alumno: Jorge Ignacio Balanz Pino             #
# Asignatura: Visión Artificial                 #
# Curso: 2016/2017                              #
# Universidad: URJC                             #
#################################################

import cv2
import os
import sys
from recognition.Recognizer import Recognizer
from transformations.ProcessedImage import ProcessedImage

class FileRecognizer:
    """
    Clase con las funciones para reconocer los objetos de un fichero de imagen,
    de un directorio con imágenes o de realizar una prueba de testing...
    """

    def recognizeImage(filename: str, classifier: Recognizer,
                       classnames: []=[], rect_colors: []=[],
                       show_marked_image: bool=True, debug: bool=False)->[]:
        """
        Reconocer una imagen. Toma una fichero de imagen e intenta reconocer los objetos
        que contenga. Retorna una lista con los códigos de clase de los objetos reconocidos.
        :param classifier: Clasificador que se utilizaré para reconocer objetos.
        :param classnames: Lista con los nombres de las clases. La posición en la lista debe coincidir con el código de la clase. 
        :param rect_colors: Lista de colores que se utilizará para rotular la imagen antes de presentarla. Un color por cada clase.
        :param show_marked_image: Indica si debe presentarse la imagen con los objetos reconocidos rotulados y remarcados.
        :param debug: Indica si se debe presentar información de depuración.
        :return: Lista con los códigos de objetos reconocidos.
        """
        try:
            if not filename or not os.path.isfile(filename):
                raise IOError("No se encuentra el fichero '{}'...".format(filename))
            # Leemos la imagen
            img = cv2.imread(filename)
            # La procesamos para extraer los descriptores
            img_proc = ProcessedImage(img)
            # Clasificamos cada patrón obtenido
            i = 0
            classifications = []
            r = []
            for pattern in img_proc.descriptors:
                features = pattern.getFeatures()
                if features:
                    pattern.classification, pattern.code = classifier.classify(pattern.getFeatures(), Recognizer.KNN_CLASSIFIER, debug=debug)
                    if debug:
                        r = pattern.getRectangle()
                        print("Clasificación del contorno '{}': posición ({},{}): Class-Code {} / Class-Name '{}'".format(
                            i, r[0], r[1], pattern.code, pattern.classification
                        ))
                    i += 1
                    classifications.append(pattern.code)
                else:
                    if debug:
                        print("[FileRecognizer] No se han podido extraer descriptores de la imagen '{}'.".format(filename))
            # Cuando procesamos todos los objetos de la imagen, lo presentamos:
            if show_marked_image:
                img_proc.markAllContours(classes=classnames, colors=rect_colors, windowText=filename)
            # Retornamos el resultado del reconocimiento:
            return classifications
        except IOError as ioe:
            sys.stderr.write("\n[FileRecognizer] I/O Error: {}.\n".format(ioe))
            return None
        except OSError as ose:
            sys.stderr.write("\n[FileRecognizer] OS Error: {}.\n".format(ose))
            return None
        except Exception as e:
            sys.stderr.write("\n[FileRecognizer] Error Detected: {}.\n".format(e))
            return None

    def recognizeDirectory(dirname: str, classifier: Recognizer, dir_class_code: int=-1,
                           classnames: [] = [], rect_colors: [] = [],
                           show_images: bool=False, debug: bool = False
                           )->(int, int, int):
        """
        Reconocer los objetos de las imágenes de todo un directorio.
        Por cada imagen, hace una llamada a la función recognizeImage, de esta misma clase.
        :param classifier: Clasificador que se utilizaré para reconocer objetos.
        :param dir_class_code: Código de directorio, para realizar un reconocimiento automático.
            Los objetos de las imágenes del directorio se presuponen como pertenecientes a 
            la clase indicada en este parámetro, a efectos de cómputos de acierto/error. 
        :param classnames: Lista con los nombres de las clases. La posición en la lista debe coincidir con el código de la clase. 
        :param rect_colors: Lista de colores que se utilizará para rotular la imagen antes de presentarla. Un color por cada clase.
        :param show_marked_image: Indica si debe presentarse la imagen con los objetos reconocidos rotulados y remarcados.
        :param debug: Indica si se debe presentar información de depuración.
        :return: (aciertos, fallos, num_ficheros). Número de objetos reconocidos correctamente, Número de objetos reconocidos incorrectamente 
            y número de ficheros procesados en el directorio.
        """
        if not dirname or not os.path.isdir(dirname):
            raise IOError("No se encuentra el directorio '{}'...".format(dirname))

        success = 0
        failures = 0
        filecounter = 0
        # Recorremos los subdirectorios:
        for base, subdir, subfiles in os.walk(dirname):
            # Mostramos algo de información:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(">>> {} <<< Reconociendo Imágenes del Directorio '{}':".format(dir_class_code, base))
            print("    Ficheros a procesar: {}".format(len(subfiles)))
            # Procesamos los ficheros del subdirectorio:
            for file in subfiles:
                (fname, fext) = os.path.splitext(file)
                if fext.lower() == ".jpg" or fext.lower() == ".png" or fext.lower() == ".bmp":
                    f = os.path.join(base, file)
                    # Reconocemos la imagen:
                    cls = FileRecognizer.recognizeImage(f, classifier, classnames, rect_colors, show_images, debug=debug)
                    if debug:
                        print("<< {} >> Clasificación del fichero '{}': {}; DirCode: {}".format(filecounter, f, cls, dir_class_code))

                    # Comprobamos si el número de directorio coincide con el número de clase
                    # Este dato es sólo útil para la práctica... Para la fase de testing automatizado
                    if cls != []:
                        for x in cls:
                            if x == dir_class_code:
                                success += 1
                            else:
                                failures += 1
                                if dir_class_code >= 0:
                                    print(
                                        "  !! [FileRecognizer] Reconocimiento incorrecto de la imagen '{}': detectado: {}; REAL {}".format(
                                        f, x, dir_class_code))
                    else:
                        print("  !! [FileRecognizer] El reconocimiento de imagen no está retornando nada para el archivo '{}': {}\n".format(f, cls))
                    filecounter += 1
                else:
                    if debug:
                        print("  > Fichero '{}' no se procesa (sólo se aceptan ficheros jpg, png y bmp).".format())
            # Fin del FOR de los ficheros
        # Fin del FOR del subdirectorio
        # Presentamos estadística:
        if dir_class_code >= 0:
            print("\n-- @@ Ficheros procesados: {}".format(filecounter))
            print("-- @@ Objetos encontrados y procesados: {}".format(success + failures))
            print("-- @@ Coincidencias: {}".format(success))
            print("-- @@ Errores: {}".format(failures))
            total = (success + failures)
            if total <= 0:
                total = 1
            print("-- @@ Porcentaje de aciertos: {}%\n".format((success / total) * 100))
        else:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if debug:
                print("-- @@ El modo de ejecución no procesa estadísticas (DirClassCode = {}).\n".format(dir_class_code))
        return success, failures, filecounter

    def recognizeTestDirectory(dirname: str, classifier: Recognizer,
                           classnames: [] = [], rect_colors: [] = [],
                           show_stats: bool=True, show_images: bool=False,
                            debug: bool = False):
        """
        Función que automatiza la fase de test del clasificador.
        Recibe un nombre de subdirectorio y realiza un reconocimiento de las imágenes de los subdirectorios.
        Las imágenes de test deben estar distribuidas en una estructura de directorios igual a la de las imágenes de entrenamiento.
        :param classifier: Clasificador que se utilizaré para reconocer objetos.
        :param classnames: Lista con los nombres de las clases. La posición en la lista debe coincidir con el código de la clase. 
        :param rect_colors: Lista de colores que se utilizará para rotular la imagen antes de presentarla. Un color por cada clase.
        :param show_stats: Indica si se debe presentar información de las estadísticas de aciertos y fallos.
        :param show_images: Indica si debe presentarse la imagen con los objetos reconocidos rotulados y remarcados.
        :param debug: Indica si se debe presentar información de depuración.
        :return: (aciertos, fallos, num_ficheros). Número de objetos reconocidos correctamente, Número de objetos reconocidos incorrectamente 
            y número de ficheros procesados en el directorio.
        """
        if not dirname or not os.path.isdir(dirname):
            raise IOError("No se encuentra el directorio '{}'...".format(dirname))

        success = 0
        failures = 0
        filecounter = 0
        dircounter = 0
        # Recorremos los subdirectorios:
        for base, directories, files in os.walk(dirname):
            # dirs va cogiendo el nombre de los subdirectorios
            for dirs in directories:
                # Ahora procesamos las imágenes de los subdirectorios
                d = os.path.join(base, dirs)
                s, f, fc = FileRecognizer.recognizeDirectory(d, classifier, dircounter, classnames, rect_colors, False, debug)
                # Ajustamos los datos para la estadística:
                success += s
                failures += f
                filecounter += fc
                # Una vez procesado el directorio, incrementamos el contador de directorios:
                dircounter += 1
        # Presentamos estadística:
        if show_stats:
            # Formateo de salida:
            # https://pyformat.info/
            print("\n-------------------------------------------------------")
            print("{:.<42}{:3d}{:>10}".format("-- @@ Ficheros procesados:",filecounter, "--"))
            print("{:.<42}{:3d}{:>10}".format("-- @@ Objetos encontrados y procesados:", success + failures, "--"))
            print("{:.<42}{:3d}{:>10}".format("-- @@ Coincidencias:", success, "--"))
            print("{:.<42}{:3d}{:>10}".format("-- @@ Errores:", failures, "--"))
            total = (success + failures)
            if total != 0:
                total = (success / total) * 100
            else:
                total = "ERROR"
            print("{:.<42}{:6.2f}%{:>6}".format("-- @@ Porcentaje de aciertos:", total, "--"))
            print("-------------------------------------------------------\n")
        return success, failures, filecounter
