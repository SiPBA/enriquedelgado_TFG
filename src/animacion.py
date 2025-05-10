###################################################################################################
# ANIMACIÓN DEL PROCESO DE ENTRENAMIENTO DE LOS MODELOS:                                          #
#--------------------------------------------------------------------------------------------------
# Este script se divide en 2 partes:                                                              #
#--------------------------------------------------------------------------------------------------
# 1. La primera parte se encarga de animar cada una de las imágenes reconstruidas en el proceso   #
# de entrenamiento (Cortes axiales, coronales y sagitales del cerebro, la función de pérdidas y   #
# la evolución de algunas variables latentes elegidas por el usuario).                            #
#--------------------------------------------------------------------------------------------------
# 2. La segunda parte se encarga de unir las animaciones creadas en la primera parte en un solo   #
# vídeo.                                                                                          #
#--------------------------------------------------------------------------------------------------
# Nota: Ejecutar en el kernel (environment) en el que se tenga instalado cv2 y moviepy (base en   # 
# mi caso).                                                                                       # 
###################################################################################################
import cv2
from tqdm import tqdm
from moviepy.editor import VideoFileClip, clips_array
###################################################################################################
#   VARIABLES DE CONTROL:                                                                         #
#   num_epoch: Número de epochs empleados en el entrenamiento.                                    #
#   d:         Dimensiones latentes del modelo entrenado.                                         #
#   modelo:    Modelo de variables latentes empleado (CAE, CVAE).                                 #
#   sujetos:   Sujetos escogidos para animar.                                                     #
#   Img_ini:   Número de la primera imagen con la que empieza el video.                           #
#   Img_end:   Número de la última imagen con la que termina el video.                            #
#   fps:       Número de imagenes por segundo.                                                    #
###################################################################################################
num_epoch = 20
d = 16
#modelo = 'CAE'
modelo = 'CVAE'
sujetos = ['3853_0','3540_4']
Img_init = 0
Img_end = 19
fps = 4
#--------------------------------------------------------------------------------------------------
###################################################################################################
#                                      INICIO DEL PROGRAMA:                                       #
###################################################################################################
animacion = ['Axial','Coronal','Sagital','FuncionPerdidas','EspacioLatente1','EspacioLatente2','EspacioLatente3']
for sujeto in sujetos:
    for corte in animacion:
        img_array = []
        print('Tarea 1/2:')
        if corte != 'FuncionPerdidas':
            for i in tqdm(range (Img_init, Img_end)):
                path = 'C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/ImagenesReconstruidas/'+str(corte)+'/Paciente_'+str(sujeto)+'_'+str(i)+'.jpg'
                img = cv2.imread(path)
                img_array.append(img)
            print('Hecho!')
        else:
            for i in tqdm(range (Img_init, Img_end)):
                path = 'C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/'+str(corte)+'/decoded'+str(i)+'.jpg'
                img = cv2.imread(path)
                img_array.append(img)
            print('Hecho!')

        height, width  = img.shape[:2]
        if corte != 'FuncionPerdidas':
            video = cv2.VideoWriter('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/'+str(corte)+'/Animacion'+str(corte)+'_'+sujeto+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
        else:
            video = cv2.VideoWriter('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/'+str(corte)+'/Animacion'+str(corte)+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
        print('Tarea 2/2:')
        for i in tqdm(range(len(img_array))):
            video.write(img_array[i])
        print('Hecho!')
        video.release()        
#--------------------------------------------------------------------------------------------------

###################################################################################################
#                             SECCIÓN PARA UNIR VIDEOS ("Stack them"):                            #
###################################################################################################
sujeto1 = sujetos[1]
sujeto2 = sujetos[0]

clip1 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/Axial/AnimacionAxial_'+sujeto1+'.mp4')
clip2 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/Coronal/AnimacionCoronal_'+sujeto1+'.mp4')
clip3 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/Sagital/AnimacionSagital_'+sujeto1+'.mp4')
clip4 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/FuncionPerdidas/AnimacionFuncionPerdidas.mp4')
clip5 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/EspacioLatente2/AnimacionEspacioLatente2_'+sujeto1+'.mp4')
clip6 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/EspacioLatente3/AnimacionEspacioLatente3_'+sujeto1+'.mp4')
clip7 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/Axial/AnimacionAxial_'+sujeto2+'.mp4')
clip8 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/Coronal/AnimacionCoronal_'+sujeto2+'.mp4')
clip9 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/Sagital/AnimacionSagital_'+sujeto2+'.mp4')
clip10 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/FuncionPerdidas/AnimacionFuncionPerdidas.mp4')
clip11 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/EspacioLatente2/AnimacionEspacioLatente2_'+sujeto2+'.mp4')
clip12 = VideoFileClip('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/Animaciones/EspacioLatente3/AnimacionEspacioLatente3_'+sujeto2+'.mp4')

final = clips_array([[clip1, clip2, clip3], [clip4, clip5, clip6], [clip7, clip8, clip9], [clip10, clip11, clip12]])
final.write_videofile('C:/TFG/Trabajo/Resultados/'+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo+'/AnimacionFinal.mp4')
#--------------------------------------------------------------------------------------------------