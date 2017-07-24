# ObjectRegognition
Application in Python and OpenCV to classifying some kinds of pills.

1 INTRODUCCIÓN:
El presente trabajo tiene por objeto identificar o clasificar imágenes de diferentes tipos de pastillas, píldoras y cápsulas 
(pastillas, en adelante) de entre una serie de tipos que se han facilitado como ejemplo. Se trata de crear una aplicación que 
a partir de unas imágenes de muestra, con pastillas de diferentes tipos, sea capaz de averiguar de qué tipo es una pastilla 
que se encuentre en una imagen y que no esté, necesariamente, entre las imágenes de muestra. De las imágenes de muestra se deben 
deducir las clases de pastillas con que se va a trabajar y las características que nos van a permitir distinguir unos tipos de 
pastillas de otros, para más tarde poder deducir de qué tipo es una pastilla a partir de su imagen.
2 OBJETIVOS A ALCANZAR
El producto de este trabajo será una aplicación capaz de realizar las siguientes operaciones:
1. Recibir una serie de imágenes de muestra de cada uno de los tipos de pastillas a reconocer, extraer sus características y 
  almacenarlas de forma adecuada para permitir su uso posterior en las identificaciones (fase de aprendizaje);
2. Recibir una serie de imágenes de muestra de cada uno de los tipos de pastillas a reconocer, cuyo tipo es conocido, 
  para comprobar el correcto funcionamiento de la aplicación (fase de test);
3. Reconocer o clasificar la imagen de una pastilla, cualquiera, a partir de los datos extraídos en la fase correspondiente 
  al punto 1.
Es importante destacar que el presente trabajo se restringe a reconocer imágenes de pastillas con características similares 
  a las imágenes que se han facilitado para las fases de aprendizaje y de test. Éstas tienen en común que las imágenes de cada 
  pastilla tienen una iluminación más o menos regular (ni demasiado luminoso ni demasiado sombrío) y que las pastillas se 
  encuentran sobre un fondo oscuro, casi negro. Los resultados obtenidos en esta práctica se basan en pruebas realizadas sobre 
  imágenes con estas características (las facilitadas para la práctica): si se utilizan otros escenarios (fondos claros, 
  iluminación demasiado alta o demasiado baja, etc…), los resultados pueden ser diferentes.
  
  3 ARCHIVOS DE LA ENTREGA
En la entrega se facilitan varios archivos distribuidos en varias carpetas.
En la raíz (el directorio raíz) se encuentran la memoria del trabajo, que incluye el MANUAL DE USO, y el ejecutable 
(archivo por lotes “.bat”) que lanza el programa en un sistema Windows que tenga instalado el software necesario 
para ejecutar programas codificados en Python (por ejemplo, Anaconda). 
Además se encuentran los siguientes directorios:
1. Carpeta “images”: Contiene las imágenes que se han facilitado para la práctica. Las carpetas cuyo nombre empieza por “Train” 
  contienen subcarpetas con imágenes para el entrenamiento, y las que empiezan por “Test” son imágenes para comprobar el 
  funcionamiento del programa (comprobar que el aprendizaje ha sido eficaz y el programa realiza su función).
2. Carpeta “PruebasSVA”: En esta carpeta se encuentran 3 archivos de muestra para comprobar el funcionamiento del programa. 
  Son imágenes con varias pastillas de diferentes tipos. La imagen “muestras_todo.jpg”, por ejemplo, contiene una imagen de 
  cada uno de los tipos de pastillas para realizar una comprobación rápida de todas las clases de pastillas.
3. Carpeta “ObjectRecognition”: Es la carpeta que contiene el programa en Python que da respuesta al problema planteado para 
  la práctica.
