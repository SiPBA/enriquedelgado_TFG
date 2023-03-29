#%% Generate models
import torch
import numpy as np 
import models
from collections import OrderedDict
import torchvision
from tqdm import tqdm

ruta = '/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/'
# ruta = 'C:\TFG\IMAGENES_TFG/'
num_epochs = 300
modelo_elegido='genvae'
d = 8
bs = 16
num_epochs=200
lr = 1E-3
norm = 3
BETA = 100
filename = f'Conv3DVAE_d{d}_BETA{int(BETA)}_lr{lr:.0E}_bs{bs}_n{num_epochs}_norm{norm}'

#%% Create model
vae = models.Conv3DVAE(encoder=models.Conv3DVAEEncoder,
                       decoder=models.Conv3DVAEDecoder,
                       latent_dim=d)
state_dict = torch.load(filename+'.pth')
new_state_dict = OrderedDict()
for k in state_dict.keys():
    nk = k.replace('_orig_mod.', '')
    new_state_dict[nk] = state_dict[k]
    
vae.load_state_dict(new_state_dict)
vae.eval()
#%% Generate Images
slic = 40
DEFAULT_Z = 0
image_size = (96, 128, 96) #2D images
values = np.arange(-5, 5, .5)
xx, yy = np.meshgrid(values, values)
input_holder = np.zeros((1, 2))
# Matrix that will contain the grid of images
container = torch.zeros((xx.size,1)+ image_size)
batch_size = 16

variables = [1, 6]
input_holder = []
for i in range(d):
    if i==variables[0]:
        input_holder.append(xx.flatten())
    elif i==variables[1]:
        input_holder.append(yy.flatten())
    else:
        input_holder.append([DEFAULT_Z]*xx.size)

input_holder = torch.Tensor(np.c_[input_holder].T)
for i in tqdm(range(0, len(input_holder), batch_size)):
    with torch.no_grad():
        output = vae.decode(input_holder[i:i+batch_size])
    container[i:i+batch_size] = output

#%% CREATE GRID
grid = torchvision.utils.make_grid(container[...,60,:,:], nrow=len(values))


#%% Show manifold
import matplotlib.pyplot as plt 
plt.figure(figsize=(20,20))
plt.imshow(grid[0], vmin=0, vmax=1)


# %%
