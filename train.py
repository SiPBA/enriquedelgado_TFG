###################################################################################################
# DEFINICIÓN DE LA FUNCIÓN DE ENTRENAMIENTO DE LOS MODELOS:                                       #
#--------------------------------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils import guarda_imag
from loader import DataLoader
import nibabel as nib
from skimage.transform import resize
from sklearn.decomposition import PCA
import pandas as pd
from utils import df_latente, anima_latente, representacion_latente
from loss import *

def train_epoch(model, dataloader, optimizer, epoch, modelo_elegido, guardar_imagenes, num_epochs, device, loss_fn, d):
    '''Función que realiza el proceso de entrenamiento en cada epoch del modelo elegido, pasando los datos por el modelo,
    calculando las pérdidas y después realizando el descenso del gradiente y la propagación hacia atrás.'''
    train_loss = []
    for image_batch in tqdm(dataloader): 
        # Mueve el tensor a la GPU si está disponible
        patno, year = image_batch[1], image_batch[2]
        image_batch = image_batch[0].to(device)
        ##############################################################################################
        #                          ELECCIÓN DEL MODELO DE ENTRENAMIENTO:                             #
        #---------------------------------------------------------------------------------------------       
        if modelo_elegido == 'CVAE':
            # Se pasan los datos por el encoder
            z, mu_latent, logvar_latent = model.encode(image_batch)
            # Se pasan los datos por el decoder
            mu_x, logvar_x = model.decode(z)
            decoded_data = mu_x
            # Ajuste de las dimensiones de la imagen reconstruida para que coincidan con las de la original 
            decoded_data = F.pad(input=decoded_data.reshape(image_batch.shape[0],1,92,124,92), pad=(0,4,0,4,0,4,0,0,0,0), mode='constant', value=0)
            mu_x = F.pad(input=mu_x.reshape(image_batch.shape[0],1,92,124,92), pad=(0,4,0,4,0,4,0,0,0,0), mode='constant', value=0)
            logvar_x = F.pad(input=logvar_x.reshape(image_batch.shape[0],1,92,124,92), pad=(0,4,0,4,0,4,0,0,0,0), mode='constant', value=0)
            # Función de perdidas  
            loss = loss_function(image_batch, mu_x, logvar_x, mu_latent, logvar_latent)
            #loss = loss_BCE(decoded_data, image_batch, mu_latent, logvar_latent)
            # Cómputo de gradientes y propagación hacia atrás
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
            if guardar_imagenes:
                guarda_imag(decoded_data, patno, year, epoch, modelo_elegido, num_epochs, d)

        elif modelo_elegido == 'CAE':
            # Se pasan los datos por el CAE
            decoded_data = model(image_batch)
            # Ajuste de las dimensiones de la imagen reconstruida para que coincidan con las de la original 
            decoded_data = F.pad(input=decoded_data, pad=(0,4,0,4,0,4,0,0,0,0), mode='constant', value=0)
            # Función de perdidas  
            loss = loss_fn(decoded_data, image_batch)
            # Cómputo de gradientes y propagación hacia atrás
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
            if guardar_imagenes:
                guarda_imag(decoded_data, patno, year, epoch, modelo_elegido, num_epochs, d)
        else:
            print('El modelo introducido no está definido')

    return np.mean(train_loss), decoded_data

#--------------------------------------------------------------------------------------------------------------
def train(modelo_elegido, train_dataset, num_epochs, model, device, optim, guardar_imagenes, animar_latente, train_loader, d):
    '''Función que realiza el entrenamiento del modelo seleccionado'''    
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
        print('------------------------------------------------------------------\n\t\tOBTENIENDO LAS IMÁGENES DEL DATASET:\n------------------------------------------------------------------')
        for image_batch in tqdm(train_loader):
            # Obtención de las imagenes, el número de sujeto y año de visita
            image = image_batch[0].numpy()
            label1 = str((image_batch[1].numpy()).astype("int16"))
            sujeto.append(label1)
            label2 = str((image_batch[2].numpy()).astype("int16"))
            year.append(label2)
            # Redimensionado de las imágenes   
            imgA = nib.Nifti1Image(resize(image[0,0,:,:,:], (91, 109, 91), preserve_range=True), np.eye(4))
            Xdata = imgA.dataobj[mask_data.astype(bool)]
            X.append(Xdata)

        X =np.vstack(X)
        pca = PCA(n_components=d)
        Z = pca.fit_transform(X)
        espacio_latente = pd.DataFrame(data = Z, columns = [f'Variable {i}' for i in range(d)])
        year = [year.strip("[]") for year in year]
        sujeto = [sujeto.strip("[]") for sujeto in sujeto]
        espacio_latente['Año'] = year
        espacio_latente['Sujeto'] = sujeto
        print('------------------------------------------------------------------\n\t\t    VARIABLES LATENTES OBTENIDAS:\n------------------------------------------------------------------')
        print(espacio_latente)
        representacion_latente(espacio_latente)
        diz_loss = []
        return diz_loss, espacio_latente

    else:
        # Entrenamiento del modelo
        loss_fn = loss_CAE()
        diz_loss = {'train_loss':[]}
        print('---------------------------------------------------------\n\t\tENTRENAMIENTO DEL MODELO:\n---------------------------------------------------------')
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1} de {num_epochs}:')
            train_loss, decoded_data = train_epoch(model, train_loader, optim, epoch, modelo_elegido, guardar_imagenes, num_epochs, device, loss_fn, d)
            if animar_latente:
                print(f'Guardando espacio latente para animación:')
                test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
                espacio_latente = df_latente(test_dataloader, model, modelo_elegido, device)
                anima_latente(espacio_latente, epoch, modelo_elegido, num_epochs, d)
            diz_loss['train_loss'].append(train_loss)
            print('Valor de la función de pérdidas: %f' % (train_loss),'\n---------------------------------------------------------')
        test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        espacio_latente = df_latente(test_dataloader, model, modelo_elegido, device)
        return diz_loss, espacio_latente
    
#--------------------------------------------------------------------------------------------------------------