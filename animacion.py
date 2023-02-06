# EJECUTAR EN EL KERNEL (ENVIRONMENT) EN EL QUE TENGA INSTALADO cv2 y moviepy (base)
import cv2
from tqdm import tqdm
################################################################################################
#   VARIABLES DE CONTROL:                                                                      #
#   Img_ini: Número de la primera imagen con la que empieza el video.                          #
#   Img_end: Número de la última imagen con la que termina el video.                           #
#   fps: Número de imagenes por segundo.                                                       #
################################################################################################
Img_init = 101
Img_end = 50000
fps = 700
################################################################################################
#                                   INICIO DEL PROGRAMA:                                       #
################################################################################################

img_array = []
n=0
m=0
print('Tarea 1/2:')
for i in tqdm(range (Img_init, Img_end)):
    path = 'C:\TFG\Codigo\SeccionReconstruidas/decoded'+str(i)+'.jpg'
    img = cv2.imread(path)
    img_array.append(img)
    n+=1
print('Hecho!')

height, width  = img.shape[:2]
video = cv2.VideoWriter('C:\TFG\Codigo\Animacion\Animacion'+str(fps)+'fps.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
print('Tarea 2/2:')
m=0
for i in tqdm(range(len(img_array))):
    video.write(img_array[i])
print('Hecho!')
video.release()        

################################################################################################
#                                   CÓDIGO PARA UNIR VIDEOS:                                   #
################################################################################################
# from moviepy.editor import *
# clip1 = VideoFileClip("C:\TFG\Codigo\Animacion\Animacion25fps.mp4")
# clip2 = VideoFileClip("C:\TFG\Codigo\Animacion\Animacion700fps.mp4")
# clips = [clip1, clip2]
# final = concatenate_videoclips(clips)
# final.write_videofile("C:\TFG\Codigo\Animacion\merged.mp4")
