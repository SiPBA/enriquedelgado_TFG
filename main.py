 #%%
import numpy as np 
import torch
import matplotlib.pyplot as plt
import plotly.express as px
from loader import ImageDataset, DataLoader
from models import *
from utils import *
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.decomposition import PCA
from skimage.transform import resize
import nibabel as nib
#from torch.utils.data import random_split
#import matplotlib.image         # Descomentar para generar las imagenes 2D reconstruidas
#import nibabel as nib           # Descomentar para generar las imagenes 3D reconstruidas 

###################################################################################################
# CONFIGURACIÓN INICIAL DEL SISTEMA:                                                              #
#--------------------------------------------------------------------------------------------------
# Eleccion del directorio de trabajo:
#-------------------------------------
# ruta = '/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/'
ruta = 'C:\TFG\IMAGENES_TFG/'
#--------------------------------------
# Eleccion del modelo de entrenamiento:                                                           
#--------------------------------------
# modelo_elegido = 'PCA'
# modelo_elegido = 'CAE'
modelo_elegido = 'CVAE'
#---------------------------------------------------------------------------------------------------
# Elección del número de epochs, dimensiones del espacio latente, learning rate y semilla aleatoria 
# para obtener resultados reproducibles:
#---------------------------------------------------------------------------------------------------
num_epochs = 2
d = 4
lr = 1e-3
torch.manual_seed(0)
###################################################################################################
print('---------------------------------------------------------\nModelo elegido:\t',modelo_elegido)

###################################################################################################
#                     PRIMERA PARTE: CARGA DE DATOS E INICIALIZACIÓN DEL MODELO                   #
###################################################################################################

## Carga de datos 
train_dataset = ImageDataset(ruta)
batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size)

## Inicialización de las funciones de pérdidas
### Para el CAE:
loss_fn = torch.nn.MSELoss()

### Para el CVAE:
def loss_function(image_batch,decoded_data, mu_x, logvar_x, mu_latent, logvar_latent, beta=1.):
    '''Función que realiza el cómputo de la función de pérdidas bajo una asunción de gaussianidad. '''
    # neg log likelihood of x under normal assumption
    # LOG_2_PI = torch.log(2.0 * torch.acos(torch.zeros(1))).item()
    # loss_rec = -torch.sum((-0.5 * LOG_2_PI + (-0.5 * logvar_x) + (-0.5 / torch.exp(logvar_x)) * (decoded_data - mu_x) ** 2.0), dim=1)
    ######################################################################################
    # Simplificación utilizada hasta solucionar la obtención correcta de mu_x y logvar_x:
    loss_rec = loss_fn(decoded_data, image_batch)
    #------------------------------------------------------------------------------------- 
    KLD = -0.5 * torch.sum(1 + logvar_latent- mu_latent.pow(2) - logvar_latent.exp(), dim=1)
    return torch.mean(loss_rec + beta * KLD)

### Inicialización de los modelos
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
    model = CVAE_3D(encoded_space_dim=d,fc2_input_dim=128) 
    params_to_optimize = [{'params': model.parameters()}]
    optim = torch.optim.Adam(params_to_optimize, lr=lr)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Dispositivo seleccionado: {device}')
    model.to(device)

else: 
    print('Dispositivo seleccionado: CPU')
#%%
###################################################################################################
#             SEGUNDA PARTE: DEFINICIÓN DE LA FUNCIÓN DE ENTRENAMIENTO DEL MODELO                 #
###################################################################################################

## Función de entrenamiento
def train_epoch(model, dataloader, optimizer, epoch, modelo_elegido, guardar_imagenes = 0):
    model.train()
    train_loss = []
    #aux = 30 # Variable para la generación de imagenes 2D. Indica los FPS a los que se va a animar
    for image_batch in tqdm(dataloader): 
        #--------------------------------------------------------------------------------------------------------------
        # Mueve el tensor a la GPU si está disponible
        patno, year = image_batch[1], image_batch[2]
        image_batch = image_batch[0].to(device)
        ##############################################################################################
        #                     ELECCIÓN DEL MODELO DE ENTRENAMIENTO:                                  #
        #---------------------------------------------------------------------------------------------       
        if modelo_elegido == 'CVAE':
            # Se pasan los datos por el encoder
            z, mu_latent, logvar_latent = model.encode(image_batch)
            # Se pasan los datos por el decoder
            decoded_data, mu_x, logvar_x = model.decode(z)
            # Ajusto las dimensiones de la imagen reconstruida para que coincidan con la original 
            decoded_data = F.pad(input=decoded_data, pad=(0,4,0,4,0,4,0,0,0,0), mode='constant', value=0)
            # Función de perdidas  
            loss = loss_function(image_batch, decoded_data, mu_x, logvar_x, mu_latent, logvar_latent)
            # Propagación hacia atrás
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())

        elif modelo_elegido == 'CAE':
            # Se pasan los datos por el CAE
            decoded_data = model(image_batch)
            # Ajusto las dimensiones de la imagen reconstruida para que coincidan con la original 
            decoded_data = F.pad(input=decoded_data, pad=(0,4,0,4,0,4,0,0,0,0), mode='constant', value=0)
            # Función de perdidas  
            loss = loss_fn(decoded_data, image_batch)
            # Cómputo de gradientes y propagación hacia atrás
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        else:
            print('El modelo introducido no está definido')
        ###############################################################################################################
        #                           CÓDIGO PARA GUARDAR LAS IMÁGENES RECONSTRUIDAS POR EL DECODER:                    #
        #--------------------------------------------------------------------------------------------------------------
        # Para guardarla en 3D:
        #--------------------------------------------------------------------------------------------------------------
        #path = 'C:/TFG/Resultados'+str(epoch)+'epochs/ImagenesReconstruidas/3D/decoded' + str(image_no)+'.nii'
        #nib.save(nib.Nifti1Image(decoded_data.cpu().detach().numpy()[0,0,:,:,:].astype('float64'), np.eye(4)), path)
        #image_no += 1
        #--------------------------------------------------------------------------------------------------------------
        # Para guardar una sección en 2D:
        #--------------------------------------------------------------------------------------------------------------
        if guardar_imagenes:
            for idx in range(1):
                #CORTE AXIAL:
                plt.imshow(decoded_data.cpu().detach().numpy()[0,0,0:91,0:108,40])
                plt.title("Reconstrucción vista desde un\n corte axial del cerebro", dict(size=15))
                #plt.text(105, -5, str(aux)+'FPS', dict(size=20), color='red')
                #################################################################################################################################################################
                # ELEGIR DIRECTORIO:           
                #plt.savefig('C:/TFG/Trabajo/Resultados5epochs/ImagenesReconstruidas/Axial/Paciente'+ str(patno.numpy()[idx])+'_'+str(year.numpy()[idx])+'_'+str(epoch)+'.jpg')
                #----------------------------------------------------------------------------------------------------------------------------------------------------------------
                plt.clf()
                # CORTE SAGITAL:
                plt.imshow(decoded_data.cpu().detach().numpy()[0,0,55,0:108,0:91])
                plt.title("Reconstrucción vista desde un\n corte sagital del cerebro", dict(size=15))
                #################################################################################################################################################################
                # ELEGIR DIRECTORIO:
                #plt.savefig('C:/TFG/Trabajo/Resultados5epochs/ImagenesReconstruidas/Sagital/Paciente'+ str(patno.numpy()[idx])+'_'+str(year.numpy()[idx])+'_'+str(epoch)+'.jpg')
                #----------------------------------------------------------------------------------------------------------------------------------------------------------------
                plt.clf()
                # CORTE CORONAL:
                plt.imshow(decoded_data.cpu().detach().numpy()[0,0,0:91,70,0:91])
                plt.title("Reconstrucción vista desde un\n corte coronal del cerebro", dict(size=15))
                #################################################################################################################################################################
                # ELEGIR DIRECTORIO:
                #plt.savefig('C:/TFG/Trabajo/Resultados5epochs/ImagenesReconstruidas/Coronal/Paciente'+ str(patno.numpy()[idx])+'_'+str(year.numpy()[idx])+'_'+str(epoch)+'.jpg')
                #----------------------------------------------------------------------------------------------------------------------------------------------------------------
                plt.clf()
        ###############################################################################################################
    return np.mean(train_loss), decoded_data

#%%
############################################################################################################
#                            TERCERA PARTE: ENTRENAMIENTO DEL MODELO                                       #
#-----------------------------------------------------------------------------------------------------------
if modelo_elegido == 'PCA':
    ############################################################################################################
    #                                            MODELO PCA                                                    #
    #-----------------------------------------------------------------------------------------------------------
    # Inicializacion de la lista X donde se almacenan lo datos linealizados de intensidad del cerebro
    X = []
    # Inicializacion de las listas donde se almacenan los datos del numero de sujeto y año de visita
    year = []
    sujeto = []
    # Carga de la mascara
    mask = nib.load('IntensityNorm_Afin_PPMI_mask.nii')
    mask_data = mask.get_fdata()
    # Iteracion sobre cada imagen del dataloader
    train_loader = DataLoader(train_dataset, batch_size=1)
    print('---------------------------------------------------------\n\t\tREALIZANDO ANÁLISIS PCA:\n---------------------------------------------------------')
    for image_batch in tqdm(train_loader):
        # Obtención de las imagenes, el número de sujeto y año de visita
        image = image_batch[0].numpy()
        label1 = str((image_batch[1].numpy()).astype("int16"))
        sujeto.append(label1)
        label2 = str((image_batch[2].numpy()).astype("int16"))
        year.append(label2)
        # Redimensionado de las imágenes   
        imgA = nib.Nifti1Image(resize(image[:,:,:,0,0], (91, 109, 91), preserve_range=True), np.eye(4))
        Xdata = imgA.dataobj[mask_data.astype(bool)]
        X.append(Xdata)

    X =np.vstack(X)
    pca = PCA(n_components=d)
    Z = pca.fit_transform(X)
    espacio_latente = pd.DataFrame(data = Z, columns = ['Variable 0','Variable 1', 'Variable 2', 'Variable 3']) #Falta automatizar para cualquier dimensión latente
    year = [year.strip("[]") for year in year]
    sujeto = [sujeto.strip("[]") for sujeto in sujeto]
    espacio_latente['Año'] = year
    espacio_latente['Sujeto'] = sujeto
    print('------------------------------------------------------------------\n\t\t    VARIABLES LATENTES OBTENIDAS:\n------------------------------------------------------------------')
    print(espacio_latente)
    representacion_latente(espacio_latente)

else:
    ## Entrenamiento del modelo
    diz_loss = {'train_loss':[]}
    print('---------------------------------------------------------\n\t\tENTRENAMIENTO DEL MODELO:\n---------------------------------------------------------')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} de {num_epochs}:')
        train_loss, decoded_data = train_epoch(model, train_loader, optim, epoch, modelo_elegido)        
        ##########################################################################################################
        #                    EVOLUCION DEL ESPACIO LATENTE PARA ANIMACION.                                       #
        ##########################################################################################################
        # PARA EL MODELO CONVOLUCIONAL:
        # print(f'Guardando espacio latente:')
        # test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        # ## Visualización del espacio latente
        # espacio_latente = []
        # for sample in tqdm(test_dataloader):
        #     img = sample[0].to(device)
        #     label1 = str((sample[1].numpy()).astype("int16"))
        #     label2 = str((sample[2].numpy()).astype("int16"))
        #     encoder.eval()
        #     if modelo_elegido == 'CVAE':
        #         with torch.no_grad():
        #             mu_latent, logvar_latent  = encoder(img)
        #             encoded_img = EncoderVAE.sample(mu_latent, logvar_latent)
        #         encoded_img = encoded_img.flatten().cpu().numpy()
        #         encoded_sample = {f"Variable {i}": enc for i, enc in enumerate(encoded_img)}  
        #         encoded_sample["Sujeto"] = label1
        #         encoded_sample["Año"] = label2
        #         espacio_latente.append(encoded_sample)

        #     elif modelo_elegido == 'CAE':
        #         with torch.no_grad():
        #             encoded_img  = encoder(img)
                    
        #         encoded_img = encoded_img.flatten().cpu().numpy()
        #         encoded_sample = {f"Variable {i}": enc for i, enc in enumerate(encoded_img)}  
        #         encoded_sample["Sujeto"] = label1
        #         encoded_sample["Año"] = label2
        #         espacio_latente.append(encoded_sample)
        #     else:
        #         print('Modelo no definido')
        # espacio_latente = pd.DataFrame(espacio_latente)
        # espacio_latente
        # # Representación del espacio latente
        # # Elección del paciente que se quiere seleccionar
        # paciente = 1024
        # # Variable 0 vs Variable 1
        # plt.scatter(espacio_latente["Variable 0"][:], espacio_latente["Variable 1"][:], alpha=0.1)
        # plt.scatter(espacio_latente["Variable 0"][paciente], espacio_latente["Variable 1"][paciente], c='r')
        # plt.xlabel('Variable 0')
        # plt.ylabel('Variable 1')
        # plt.xlim([-3000, 2200])
        # plt.ylim([-3500, 1400])
        # plt.title("Evolución del sujeto" + str(espacio_latente["Sujeto"][paciente]))
        # ####################################################################################################################################################################
        # # ELEGIR DIRECTORIO:
        # # plt.savefig('C:/TFG/Trabajo/Resultados500epochs/ImagenesReconstruidas/EspacioLatente1/Paciente_'+ str(espacio_latente["Patient"][paciente])+"_"+str(epoch)+'.jpg')
        # #-------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # plt.clf()
        # # Variable 1 vs Variable 2
        # plt.scatter(espacio_latente["Variable 1"][:], espacio_latente["Variable 2"][:], alpha=0.1)
        # plt.scatter(espacio_latente["Variable 1"][paciente], espacio_latente["Variable 2"][paciente], c='r')
        # plt.xlabel('Variable 1')
        # plt.ylabel('Variable 2')
        # plt.xlim([-3000, 2200])
        # plt.ylim([-2500, 2400])
        # plt.title("Evolución del sujeto" + str(espacio_latente["Sujeto"][paciente]))
        # ####################################################################################################################################################################
        # # ELEGIR DIRECTORIO:
        # #plt.savefig('C:/TFG/Trabajo/Resultados500epochs/ImagenesReconstruidas/EspacioLatente2/Paciente_'+ str(espacio_latente["Patient"][paciente])+"_"+str(epoch)+'.jpg')
        # #-------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # plt.clf()
        diz_loss['train_loss'].append(train_loss)
        print('Valor de la función de pérdidas: %f' % (train_loss),'\n---------------------------------------------------------')

## Evolución de la función de pérdidas
if modelo_elegido != 'PCA':
    plt.figure(figsize=(10,8))
    plt.semilogy(diz_loss['train_loss'], label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Pérdidas')
    plt.grid()
    plt.legend()
    plt.title('Función de pérdidas')
    plt.show()
    ##########################################################################################################
    # Guardar el modelo entrenado:                                                                           #
    #---------------------------------------------------------------------------------------------------------
    #import os 
    #os.mkdir('C:\TFG\Trabajo\Resultados'+str(epoch)+'epochs')
    #torch.save(model, 'C:\TFG\Trabajo\Resultados'+str(epoch)+'epochs\ModeloEntrenado\Trained_model.pth')
    ##########################################################################################################
    #%%
    ##########################################################################################################
    #                   CUARTA PARTE: REPRESENTACIÓN DEL ESPACIO LATENTE.                                    #
    ##########################################################################################################
    test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    ## Visualización del espacio latente
    espacio_latente = []
    print('---------------------------------------------------------\n\t\tMOSTRANDO ESPACIO LATENTE\n---------------------------------------------------------')
    for sample in tqdm(test_dataloader):
        img = sample[0].to(device)
        label1 = str((sample[1].numpy()).astype("int16"))
        label2 = str((sample[2].numpy()).astype("int16"))
        model.eval()

        if modelo_elegido == 'CVAE':
                with torch.no_grad():
                    encoded_img, _, _  = model.encode(img)
                df_latente(encoded_img, label1, label2, espacio_latente)

        elif modelo_elegido == 'CAE':
            with torch.no_grad():
                encoded_img  = model.encode(img)      
            df_latente(encoded_img, label1, label2, espacio_latente)

        else:
            print('Modelo no definido')

    espacio_latente = pd.DataFrame(espacio_latente)
    print('------------------------------------------------------------------\n\t\t    VARIABLES LATENTES OBTENIDAS:\n------------------------------------------------------------------')
    print(espacio_latente)
    #Representación del espacio latente
    representacion_latente(espacio_latente)


########################################################################################################
# Código para guardar la función de pérdidas para hacer una animación de su evolución                  #
#-------------------------------------------------------------------------------------------------------
# a=np.ones(500).astype('int16')
# for i in tqdm(range(500)):
#     plt.figure(figsize=(6.4,4.8)) # Para que coincidan las dimensiones para la animación.
#     plt.semilogy(diz_loss['train_loss'][0:i], label='Train')
#     plt.xlabel('Epoch')
#     plt.ylabel('Pérdidas')
#     plt.grid()
#     plt.xlim([-10, 500])
#     plt.ylim([6e-3, 0.3])
#     plt.legend()
#     plt.title('Función de pérdidas')
#     plt.savefig('C:/TFG/Trabajo/Resultados500epochs/FuncionPerdidas/decoded'+ str(i)+'.jpg')
#     plt.clf()
#-------------------------------------------------------------------------------------------------------
