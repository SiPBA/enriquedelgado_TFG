 #%%
import numpy as np 
import torch
import matplotlib.pyplot as plt
import plotly.express as px
from loader import ImageDataset, DataLoader
from models import Encoder, Decoder
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
#from torch.utils.data import random_split
#import matplotlib.image         # Descomentar para generar las imagenes 2D reconstruidas
#import nibabel as nib           # Descomentar para generar las imagenes 3D reconstruidas 
#--------------------------------------------------------------------------------------------------

#ruta = '/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/'
ruta = 'C:\TFG\IMAGENES_TFG/'

###################################################################################################
#                     PRIMERA PARTE: CARGA DE DATOS E INICIALIZACIÓN DEL MODELO                   #
###################################################################################################

##Carga de datos 
train_dataset = ImageDataset(ruta)
batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size)

##Inicialización de la función de pérdidas y el optimizador:
loss_fn = torch.nn.MSELoss()
### Definición del learning rate
lr = 1e-3
### Semilla aleatoria para obtener resultados reproducibles
torch.manual_seed(0)
### Inicialización de las 2 redes neuronales y fijación de la dimensión del espacio latente
d = 4
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128) 
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
params_to_optimize = [ 
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
optim = torch.optim.Adam(params_to_optimize, lr=lr)

# Comprobación de si la GPU está disponible
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Dispositivo seleccionado: {device}')

# Se mueven el encoder y el decoder a la GPU si está disponible
encoder.to(device)
decoder.to(device)
#%%
###################################################################################################
#             SEGUNDA PARTE: DEFINICIÓN DE LA FUNCIÓN DE ENTRENAMIENTO DEL MODELO                 #
###################################################################################################

## Función de entrenamiento
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer, image_no):

    encoder.train()
    decoder.train()
    train_loss = []

    for image_batch in tqdm(dataloader): 
        # Mueve el tensor a la GPU si está disponible
        image_batch = image_batch[0].to(device)
        # Se pasan los datos por el encoder
        encoded_data = encoder(image_batch)
        # Se pasan los datos por el decoder
        decoded_data = decoder(encoded_data)
        # Ajusto las dimensiones de la imagen reconstruida para que coincidan con la original 
        decoded_data = F.pad(input=decoded_data, pad=(0,4,0,4,0,4,0,0,0,0), mode='constant', value=0)
        #############################################################################################################################################
        #                           CÓDIGO PARA GUARDAR LAS IMÁGENES RECONSTRUIDAS POR EL DECODER:                                                  #
        #--------------------------------------------------------------------------------------------------------------------------------------------
        #path = 'C:/TFG/codigo/Reconstruidas/decoded' + str(image_no)+'.nii'
        #--------------------------------------------------------------------------------------------------------------------------------------------
        # Para guardarla en 3D:
        #--------------------------------------------------------------------------------------------------------------------------------------------
        #nib.save(nib.Nifti1Image(decoded_data.cpu().detach().numpy()[0,0,:,:,:].astype('float64'), np.eye(4)), path)
        #image_no += 1
        #--------------------------------------------------------------------------------------------------------------------------------------------
        # Para guardar una sección en 2D:
        #--------------------------------------------------------------------------------------------------------------------------------------------
        #plt.imshow(decoded_data.cpu().detach().numpy()[0,0,:,:,40])
        #plt.title("Reconstrucción de las imágenes\n conforme avanza el entrenamiento")
        #plt.savefig('C:/TFG/codigo/SeccionReconstruidas/decoded'+ str(image_no)+'.jpg')
        #plt.clf()
        #matplotlib.image.imsave('C:/TFG/codigo/SeccionReconstruidas/decoded'+ str(image_no)+'.png', decoded_data.cpu().detach().numpy()[0,0,:,:,40])
        #image_no += 1
        ##############################################################################################################################################
        loss = loss_fn(decoded_data, image_batch)
        # Propagación hacia atrás
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Imprime las pérdidas de cada batch
        #print('\t Pérdidas parciales de entrenamiento (en cada batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss), decoded_data, image_no

#%%
###################################################################################################
#        TERCERA PARTE: ENTRENAMIENTO DEL MODELO Y REPRESENTACIÓN DE LA FUNCIÓN DE PÉRDIDAS       #
###################################################################################################

## Entrenamiento del modelo
num_epochs = 5
diz_loss = {'train_loss':[]}
image_no = 0 #Iterador usado para numerar las imágenes reconstruidas por el decoder
print('---------------------------------------------------------\n\t\tENTRENAMIENTO DEL MODELO:\n---------------------------------------------------------')

for epoch in range(num_epochs):
   print(f'Epoch {epoch+1} de {num_epochs}:')
   train_loss, decoded_data, image_no = train_epoch(encoder,decoder,device,train_loader,loss_fn,optim, image_no)
   diz_loss['train_loss'].append(train_loss)
   print('Valor de la función de pérdiads: %f' % (train_loss),'\n---------------------------------------------------------')

## Evolución de la función de pérdidas
plt.figure(figsize=(10,8))
plt.semilogy(diz_loss['train_loss'], label='Train')
plt.xlabel('Epoch')
plt.ylabel('Pérdidas')
plt.grid()
plt.legend()
plt.title('Función de pérdidas')
plt.show()

#%%
###################################################################################################
#                   CUARTA PARTE: REPRESENTACIÓN DEL ESPACIO LATENTE.                             #
###################################################################################################

test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

## Visualización del espacio latente
encoded_samples = []
n = 0
for sample in tqdm(test_dataloader):
    img = sample[0].to(device)
    label1 = (sample[1].numpy())[0,:].astype("int16")
    label2 = (sample[2].numpy())[0,:].astype("int16")
    encoder.eval()
    with torch.no_grad():
        encoded_img  = encoder(img)

    encoded_img = encoded_img.flatten().cpu().numpy()
    encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}  
    encoded_sample["Patient"] = label1[n]
    encoded_sample["Year"] = label2[n]
    encoded_samples.append(encoded_sample)
    n += 1
encoded_samples = pd.DataFrame(encoded_samples)
encoded_samples

#Representación del espacio latente
px.scatter(encoded_samples, x='Enc. Variable 0', y='Enc. Variable 1', opacity=0.3)

# %%
