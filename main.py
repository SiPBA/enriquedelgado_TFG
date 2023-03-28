#%%
###################################################################################################
# CONFIGURACIÓN DEL SISTEMA, ENTRENAMIENTO Y REPRESENTACIÓN DEL ESPACIO LATENTE:                  #
#--------------------------------------------------------------------------------------------------
# Este script se divide en cuatro partes:                                                         #
#--------------------------------------------------------------------------------------------------
# 1. Configuración inicial del sistema: En esta parte se selecciona la configuración del sistema  #
# incluyendo la ruta de la base de datos, el modelo que se quiera seleccionar para entrenar, el   #
# número de epochs, dimensiones latentes, learning rate, semilla aleatoria y opciones para        #
# guardar los resultados obtenidos.                                                               #
#--------------------------------------------------------------------------------------------------
# 2. Carga de datos e inicialización del modelo.                                                  #
#--------------------------------------------------------------------------------------------------
# 3. Entrenamiento del modelo.                                                                    #
#--------------------------------------------------------------------------------------------------
# 3. Representación del espacio latente.                                                          #
#--------------------------------------------------------------------------------------------------
import torch
from loader import ImageDataset, ImageDatasetNuevo, HDFImageDataset, DataLoader
from models import *
from utils import *
from train import *
# from image_norms import integral_norm

###################################################################################################
# CONFIGURACIÓN INICIAL DEL SISTEMA:                                                              #
#--------------------------------------------------------------------------------------------------
# Eleccion del directorio de trabajo:
#-------------------------------------
ruta = '/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/'
# ruta = 'C:\TFG\IMAGENES_TFG/'
#--------------------------------------
# Eleccion del modelo de entrenamiento:                                                           
#--------------------------------------
# modelo_elegido = 'PCA'
# modelo_elegido = 'CAE'
modelo_elegido = 'CVAE'
#---------------------------------------------------------------------------------------------------
# Elección del número de epochs, dimensiones del espacio latente, learning rate, imágenes por lote,
# semilla aleatoria para obtener resultados reproducibles y una variable lógica para guardar los
# resultados:
#---------------------------------------------------------------------------------------------------
num_epochs = 450
d = 3
lr = 1e-4
batch_size = 64
torch.manual_seed(0)
guardar_modelo_entrenado = 1
#---------------------------------------------------------------------------------------------------
# Parámetro para realizar una animación del proceso de entrenamiento de los modelos CVAE y CAE 
# (Omitir en PCA)
animar_latente = 1
guardar_imagenes = 1
###################################################################################################
print('---------------------------------------------------------\nModelo elegido:\t',modelo_elegido)

###################################################################################################
#                     PRIMERA PARTE: CARGA DE DATOS E INICIALIZACIÓN DEL MODELO                   #
###################################################################################################

# Carga de datos 
# train_dataset = ImageDataset(ruta)
train_dataset = HDFImageDataset('medical_images.hdf5')
# train_dataset = ImageDatasetNuevo(ruta, norm=integral_norm, normkws={'method': 'gmm'})
train_loader = DataLoader(train_dataset, batch_size=batch_size)

# Inicialización de los modelos
if modelo_elegido == 'CAE':
    model = CAE_3D(encoded_space_dim=d,fc2_input_dim=128) 
    params_to_optimize = [{'params': model.parameters()}]
    # Optimizador elegido: ADAM
    optim = torch.optim.Adam(params_to_optimize, lr=lr)
    # Comprobación de si la GPU está disponible
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Dispositivo seleccionado: {device}')
    # Se mueve el modelo a la GPU si está disponible
    model.to(device)

elif modelo_elegido == 'CVAE':
    # opt_model = CVAE_3D(encoded_space_dim=d,fc2_input_dim=128) 
    opt_model = Conv3DVAE(latent_dim = d)
    model = torch.compile(opt_model)
    # model = opt_model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Dispositivo seleccionado: {device}')
    model = model.to(device)
    params_to_optimize = [{'params': model.parameters()}]
    optim = torch.optim.Adam(params_to_optimize, lr=lr)
    # optim = torch.optim.SGD(params_to_optimize, lr=lr)#, momentum=0.9)

else:
    model = 0
    device = torch.device("cpu")
    optim = 0 
    print('Dispositivo seleccionado:', device)

#%%
############################################################################################################
#                            SEGUNDA PARTE: ENTRENAMIENTO DEL MODELO                                       #
#-----------------------------------------------------------------------------------------------------------
diz_loss, espacio_latente = train(modelo_elegido, train_dataset, num_epochs, model, device, optim, guardar_imagenes, animar_latente, train_loader, d)
# Guardar el modelo entrenado:                                                                           
if guardar_modelo_entrenado:
    #import os 
    #os.mkdir('C:\TFG\Trabajo\Resultados'+str(num_epochs)+'epochs')
    if modelo_elegido != 'PCA':
        torch.save(model, str(d)+'_dimensiones_latentes/'+str(num_epochs)+'epochs/'+modelo_elegido+'/ModeloEntrenado/modelo_entrenado.pth')
        espacio_latente.to_csv(str(d)+'_dimensiones_latentes/'+str(num_epochs)+'epochs/'+modelo_elegido+'/ModeloEntrenado/variables_latentes.csv')
    else: 
        espacio_latente.to_csv(str(d)+'_dimensiones_latentes/'+modelo_elegido+'/variables_latentes.csv')

#%%
############################################################################################################
#                         TERCERA PARTE: REPRESENTACIÓN DEL ESPACIO LATENTE.                               #
############################################################################################################
if modelo_elegido != 'PCA':
    # Evolución de la función de pérdidas
    representa_perdidas(diz_loss, modelo_elegido)
    # Visualización del espacio latente
    print('---------------------------------------------------------\n\t\tMOSTRANDO ESPACIO LATENTE\n---------------------------------------------------------')
    test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    espacio_latente = df_latente(test_dataloader, model, modelo_elegido, device)
    print('------------------------------------------------------------------\n\t\t    VARIABLES LATENTES OBTENIDAS:\n------------------------------------------------------------------')
    print(espacio_latente)
    # Representación del espacio latente
    representacion_latente(espacio_latente)

    ########################################################################################################
    # Código para guardar la función de pérdidas para hacer una animación de su evolución                  #
    #-------------------------------------------------------------------------------------------------------
    if guardar_imagenes:
        guarda_perdidas(diz_loss, modelo_elegido, num_epochs, d)
    #-------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------