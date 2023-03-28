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
d = 3
lr = 1e-3 # 3e-4
batch_size = 16
PARAM_BETA = 100.
PARAM_LOGSIGMA = np.log(.1)
PARAM_NORM = 3
filename = f'Conv3DVAE_d{d}_BETA{int(PARAM_BETA)}_lr{lr:.0E}_bs{batch_size}_n{num_epochs}_norm{PARAM_NORM}'

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

input_holder = torch.Tensor(np.c_[xx.flatten(), [DEFAULT_Z]*xx.size, yy.flatten()])
for i in tqdm(range(0, len(input_holder), batch_size)):
    with torch.no_grad():
        output = vae.decode(input_holder[i:i+batch_size])
    container[i:i+batch_size] = output

#%% CREATE GRID
grid = torchvision.utils.make_grid(container[..., slic], nrow=len(values))


#%% Show manifold
import matplotlib.pyplot as plt 
plt.figure(figsize=(20,20))
plt.imshow(grid[0], vmin=0.5, vmax=1)
# %% VISUALIZE CORRELATIONS
import seaborn as sns
import pandas as pd 
from sklearn.decomposition._factor_analysis import _ortho_rotation

espacio_latente = pd.read_csv(filename+'.csv')
espacio_rotated = espacio_latente.copy()
espacio_rotated[[f'Variable {i}' for i in range(d)]] = _ortho_rotation(espacio_latente[[f'Variable {i}' for i in range(d)]].values).T
# espacio_latente = df_latente(test_dataloader, vae, modelo_elegido, device)

g = sns.clustermap(espacio_latente.corr(), center=0, cmap="vlag",
                   dendrogram_ratio=(.1, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75, figsize=(12, 13))

# %%
