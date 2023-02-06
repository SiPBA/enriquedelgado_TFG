# **REGISTRO DE TRABAJO** #

## **Seguimiento semanal del trabajo realizado** ##
* * *
* * *
### ***SEMANA 1 (15/09/22):*** ###

+ Elección de tutor del TFG y **primera reunión.**
+ Descarga, instalación y documentación del Software que se va a emplear.
    + Documentación sobre el uso de _VSCode_
    + Repaso del lenguaje de programación _Python_
    + Repaso de la librería _NumPy._
* * *

### ***SEMANA 2 (19/09/22):*** ###
+ Documentación sobre redes neuronales
+ Realización de tutoriales de librerías de _Python._
    + Documentación y realización de tutoriales sobre la libería _Pytorch_ para aprendizaje profundo.
    + Documentación sobre las librerías _Nilearn_ y _Nibabel_ para procesado de imágenes médicas.
* * *

### ***SEMANA 3 (03/10/22):*** ###
+ **Segunda reunión.** Elección del tema definitivo del trabajo.
+ Descarga de todos los datos que se van a emplear (Imágenes de pacientes + bases de datos).
+ Creación de cuenta de _GitHub_ y documentación sobre _GIT_.
* * *

### ***SEMANA 4 (10/10/22):*** ###
+ Documentación incial sobre anatomía y biología cerebral, enfermedad de Parkinson y empleo de _DaTSCAN_
+ Documentación sobre la creación de _datasets_ y _dataloaders_.
+ Creación del archivo _loaders.py_ y definición de la arquitectura del cargador de datos para las imágenes cerebrales que se van a utilizar en el entrenamiento de las redes neuronales.
* * *

### ***SEMANA 5 (17/10/22):*** ###
+ **Tercera reunión.** 
+ Documentación sobre redes neuronales convolucionales.
+ Documentación sobre _Autoencoders_ y _Autoencoders_ convolucionales.
* * * 

### ***SEMANA 6 (24/10/22):*** ###
+ Realización de tutoriales sobre la creación de _Autoencoders_ convolucionales para clasificación de números.
+ Documentación sobre programación orientada a objetos en _Python._
* * *

### ***SEMANA 7 (31/10/22):*** ###
+ Creación del archivo _main.py_ 
    + Definición de la carga de datos e inicialización del modelo.
    + Definición de la función de entrenamiento.
+ Creación del archivo _models.py_ y definición de la arquitectura del primer modelo de _Autoencoder_ convolucional.
    + Definición de la arquitectura del _encoder._
    + Definición de la arquitectura del _decoder._
* * *

### ***SEMANA 8 (07/11/22):*** ###
+ Depuración del código, ajuste de parámetros y corrección de errores.
+ Entrenamiento del modelo.
* * *

### ***SEMANA 9 (14/11/22):*** ###
+ **Cuarta reunión.** Correción de errores.
+ Realización de tutorial sobre la librería _Pandas_ para análisis de datos.
+ Documentación sobre el lenguaje de marcado _Marckdown_ y creación del registro de trabajo.
+ Modificación del archivo _loaders.py_ para que cargue las imágenes y dos nuevas variables (Número de paciente y año de la visita) a través del archivo _dataset_nuevo.csv_. 
+ Adición de código para visualizar el espacio latente.
+ Adición de código en la función de entrenamiento para guardar las imágenes reconstruidas en cada paso del entrenamiento.
+ Entrenamiento del modelo con 340 epochs.
+ Creación del archivo _animacion.py_ para crear una animación con la evolución de las imágenes reconstruidas durante el entrenamiento.

* * *

### ***SEMANA 10 (21/11/22):*** ###
+ Modificación del código para guardar las imágenes reconstruidas en cada paso del entrenamiento.
    + Adición de código para guardar las imágenes correspondientes a un sujeto y una visita en concreto. 
* * *

### ***SEMANA 11 (28/11/22):*** ###
+ Adición de código en la función de entrenamiento del modelo para guardar las imágenes de la evolución del espacio latente en cada epoch. Representación de la situación de un sujeto respecto al resto.
+ Depuración del código y corrección de errores.
+ Entrenamiento del modelo para 495 epochs.
* * *

### ***SEMANA 12 (05/12/22):*** ###
+ Modificación del archivo _animacion.py_ para crear la animación del entrenamiento con un panel complejo de 6 animaciones, correspondientes a la visualización de la reconstrucción de las imágenes 3D desde tres cortes distintos, la evolución de la función de pérdidas y la del espacio latente.
+ Corrección de errores y optimización del código.
+ Entrenamiento del modelo para 500 epochs, guardando las imágenes correspondientes a otro sujeto para luego realizar una comparación. 
* * *

### ***SEMANA 13 (12/12/22):*** ###
+ Modificación del archivo _animacion.py_ para agrandar el panel y que represente 12 animaciones correspondientes a la evolución del entrenamiento del modelo para dos sujetos diferentes. 
+ Creación de una animación comparando a dos sujetos diferentes y evaluación del resultado.
* * *

### ***SEMANA 14 (19/12/22):*** ###
+ Documentación sobre la biblioteca Sciikit-learn y sobre la clase _sklearn.decomposition.PCA_
+ Documentación e implementación del análisis PCA.

* * *

### ***SEMANA 15 (26/12/22):*** ###
+ Corrección de errores del PCA.
+ Documentación sobre autoencoders variacionales y el cómputo de la función de pérdidas.
* * *

### ***SEMANA 16 (02/01/23):*** ###
+ Implementación del modelo variacional (CVAE).
+ Corrección de errores.
* * *

### ***SEMANA 17 (09/01/23):*** ###
+ Evaluación ordinaria del primer cuatrimestre
* * *

### ***SEMANA 18 (16/01/23):*** ###
+ Evaluación ordinaria del primer cuatrimestre
* * *

### ***SEMANA 19 (23/01/23):*** ###
+ Evaluación ordinaria del primer cuatrimestre
* * *
### ***SEMANA 20 (30/01/23):*** ###
+ Creación del archivo _utils.py_ con funciones para crear un dataframe con los valores de las variables latentes asociadas a cada sujeto y número de visita, y para realizar la representación del espacio latente.
+ Modificación del archivo _models.py_ para optimizarlo y que sea más legible.
+ Optimización del código del archivo _main.py_
