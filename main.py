import numpy as np 
import torch
import matplotlib.pyplot as plt
import plotly.express as px
from torch.utils.data import random_split
from torchvision import transforms
from loader import ImageDataset, DataLoader
from models import Encoder, Decoder
from torch import nn
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
import torch.nn.functional as F

###################################################################################################
#                     PRIMERA PARTE: CARGA DE DATOS E INICIALIZACIÓN DEL MODELO                   #
###################################################################################################

##Carga de datos 
train_dataset = ImageDataset()
train_transform = transforms.Compose([transforms.ToTensor(),])
train_dataset.transform = train_transform

batch_size=8 #No lo he probado con más porque no tengo suficiente RAM en mi GPU
train_loader = DataLoader(train_dataset, batch_size=batch_size)

##Inicialización de la función de pérdidas y el optimizador:
loss_fn = torch.nn.MSELoss()
### Definición del learning rate
lr = 0.001
### Semilla aleatoria para obtener resultados reproducibles
torch.manual_seed(0)
### Inicialización de las 2 redes neuronales y fijación de la dimensión del espacio latente
d = 4
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128) # No se para que se usa fc2_input_dim
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
params_to_optimize = [ #No entiendo que hace esto
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Comprobación de si la GPU está disponible
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Se mueven el encoder y el decoder a la GPU si está disponible
encoder.to(device)
decoder.to(device)

###################################################################################################
#             SEGUNDA PARTE: DEFINICIÓN DE LA FUNCIÓN DE ENTRENAMIENTO DEL MODELO                 #
###################################################################################################

## Función de entrenamiento
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):

    encoder.train()
    decoder.train()
    train_loss = []
    # Iteración del dataloader 
    for image_batch in dataloader: 
        # Mueve el tensor a la GPU si está disponible
        image_batch = image_batch.to(device)
        # Se pasan los datos por el encoder
        encoded_data = encoder(image_batch)
        # Se pasan los datos por el decoder
        decoded_data = decoder(encoded_data)
        # (Posible error en models.py) Ajusto los datos del decoder porque las dimensiones no coinciden con las originales 
        decoded_data = F.pad(input=decoded_data, pad=(0,4,0,4,0,4,0,0,0,0), mode='constant', value=0)
        # Evaluación de las pérdidas
        loss = loss_fn(decoded_data, image_batch)
        # Propagación hacia atrás
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Imprime las pérdidas de cada batch
        print('\t Pérdidas parciales de entrenamiento (en cada batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

###################################################################################################
#        TERCERA PARTE: ENTRENAMIENTO DEL MODELO Y REPRESENTACIÓN DE LA FUNCIÓN DE PÉRDIDAS       #
###################################################################################################

## Entrenamiento del modelo
num_epochs = 5 # Selecciono solo 5 epochs para que no tarde mucho
diz_loss = {'train_loss':[]}
for epoch in range(num_epochs):
   train_loss = train_epoch(encoder,decoder,device,train_loader,loss_fn,optim)
   diz_loss['train_loss'].append(train_loss)


## Evolución de la función de pérdidas
plt.figure(figsize=(10,8))
plt.semilogy(diz_loss['train_loss'], label='Train')
plt.xlabel('Epoch')
plt.ylabel('Pérdidas medias')
plt.grid()
plt.legend()
plt.title('Pérdidas')
plt.show()

###################################################################################################
#   CUARTA PARTE: REPRESENTACIÓN DEL ESPACIO LATENTE. ESTA PARTE NO HA SIDO ESTUDIADA TODAVIA     #
###################################################################################################

## Visualización del espacio latente
encoded_samples = []
for sample in tqdm(train_loader):
    img = sample[0].unsqueeze(0).to(device)
    label = sample[1]
    # Encode image
    encoder.eval()
    with torch.no_grad():
        encoded_img  = encoder(img)
    # Append to list
    encoded_img = encoded_img.flatten().cpu().numpy()
    encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
    encoded_sample['label'] = label
    encoded_samples.append(encoded_sample)
encoded_samples = pd.DataFrame(encoded_samples)
encoded_samples

# Representación del espacio latente
px.scatter(encoded_samples, x='Enc. Variable 0', y='Enc. Variable 1', 
           color=encoded_samples.label.astype(str), opacity=0.7)

#Reducción de dimensión t-SNE
# tsne = TSNE(n_components=2)
# tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
# fig = px.scatter(tsne_results, x=0, y=1,
#                  color=encoded_samples.label.astype(str),
#                  labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
# fig.show()