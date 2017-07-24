# -*- coding: utf-8 -*-
#################################################
# Alumno: Jorge Ignacio Balanz Pino             #
# Asignatura: Visión Artificial                 #
# Curso: 2016/2017                              #
# Universidad: URJC                             #
#################################################

import cv2
import sys
import os
import random
from training.Training import Training
from transformations.ProcessedImage import ProcessedImage
from recognition.Recognizer import Recognizer
from recognition.FileRecognizer import FileRecognizer


def showMenu():
    """ Opciones del menú """
    print ("\n******************************************************************")
    print (" 1.- Realizar Aprendizaje del clasificador (Automático).")
    print (" 2.- Realizar Aprendizaje del clasificador (Manual).")
    print (" 3.- Entrenar Clasificador y Realizar Test (Automático).")
    print (" 4.- Reconocer los objetos de una imagen (se muestra la imagen).")
    print (" 5.- Reconocer un directorio completo (se muestra cada imagen).")
    print (" 0.- SALIR!")
    print ("******************************************************************\n")


def create_random_colors(num: int):
    # Valores aleatorios entre 100 y 255 (a partir de 100 para que sean más bien claros)
    return [(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for i in range(num)]


def main():
    """ Presentar un menú de opciones para probar las diferentes funciones """
    image = None
    train: Training = None
    recogn: Recognizer = None
    classnames = []
    classcodes = []
    classcolors = []
    op = -1
    while (op != '0'):
        try:
            showMenu()
            op = input("\t Seleccione una opción:\t")

            if op == '1': # Aprendizaje del Clasificador AUTOMÁTICO
                classnames = []
                classcolors = []
                dirname = input("\nIntroduzca el directorio con imágenes de entrenamiento:  ")
                # Creamos un objeto Training ("entrenador")
                train = Training("AutoTrain_"+dirname, dirname, Training.AUTO_TRAINING)
                # Le hacemos aprender y retorna un clasificador (tipo KNN)
                # los nombres de las clases generados y los códigos de las clases asociados:
                recogn, classnames, classcodes = train.learning()
                # Si tenemos la lista de clases, creamos colores asociados (aleatorios):
                if classnames != []:
                    classcolors = create_random_colors(len(classnames))

            elif op == '2': # Aprendizaje del Clasificador MANUAL
                # Lo mismo que la opción 1, pero para cada subdirectorio pide un nombre de clase...
                classnames = []
                classcolors = []
                dirname = input("\nIntroduzca el directorio con imágenes de entrenamiento:  ")
                train = Training("ManualTrain_"+dirname, dirname, Training.MANUAL_TRAINING)
                recogn, classnames, classcodes = train.learning()
                # Si tenemos la lista de clases, creamos colores asociados:
                if classnames != []:
                    classcolors = create_random_colors(len(classnames))

            elif op == '3': # Entrenar Clasificador y Realizar Test (Automático).
                dirname = input("\nIntroduzca el directorio con imágenes de ENTRENAMIENTO:  ")
                # Creamos un entrenador
                train = Training("Recogn_"+dirname, dirname, Training.AUTO_TRAINING)
                # Al entrenarlo se genera un reconocedor (clasificador),
                #  una lista de nombres de clase y una lista de códigos de clase:
                recogn, classnames, classcodes = train.learning()
                # Si tenemos la lista de clases, creamos colores asociados:
                if classnames:
                    classcolors = create_random_colors(len(classnames))
                    # Ahora se pide un directorio con imágenes de test, para
                    # realizar la fase de Testing:
                    dirname = input("\nIntroduzca el directorio con imágenes de TEST:  ")
                    FileRecognizer.recognizeTestDirectory(dirname, recogn, classnames, classcolors)
                else:
                    raise Exception("No se ha conseguido procesar ningún patrón.")

            elif op == '4': # Reconocer los objetos de una imagen
                if recogn is None:
                    raise Exception("El clasificador aún no ha sido creado ni entrenado.")
                clsname = input("\nIntroduzca un nombre del fichero de imagen:  ")
                FileRecognizer.recognizeImage(clsname, recogn, classnames, classcolors)

            elif op == '5': # Reconocer un directorio completo (se muestra cada imagen)
                if recogn is None:
                    raise Exception("El clasificador aún no ha sido creado ni entrenado.")
                dirname = input("\nIntroduzca el directorio con imágenes para reconocer:  ")
                FileRecognizer.recognizeDirectory(dirname, recogn, -1, classnames, classcolors, show_images=True)

            else:
                print("\nLa opción '{}' no está disponible...\n".format(op))

            # FIN de los IF-ELIF
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            sys.stderr.write("\n[ObjectRecognition] ERROR: {}".format(e))
            sys.stderr.write("\n[ObjectRecognition] Exception: {0};\nFile: {1};\nLine: {2}.\n".format(exc_type, fname, exc_tb.tb_lineno))

        if not (op in ('0','')):
            input("\n ...Pulse <ENTER> para continuar...")

    print("\n\n***********  FIN DEL PROGRAMA  **********\n")

# PROGRAMA PRINCIPAL:

# Mostramos el menú:
if __name__ == '__main__':
	main()
