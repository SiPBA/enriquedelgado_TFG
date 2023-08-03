###################################################################################################
# CARGADOR DE LOS DATOS PARA EL ENTRENAMIENTO Y ANÁLISIS DE LOS RESULTADOS.                       #
#--------------------------------------------------------------------------------------------------
import numpy as np 
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F
from image_norms import integral_norm
import h5py
#import matplotlib.pyplot as plt

# DATASET:
class ImageDataset(Dataset):
    
    def __init__(self, ruta='C:/TFG/IMAGENES_TFG/', norm=50):   
        """_summary_
            UPDATE 16/03: pakitochus: He añadido el norm en el 
            constructor de la clase, de manera que luego podamos
            modificar (como he hecho) la clase 
        Args:
            ruta (str, optional): _description_. Defaults to 'C:/TFG/IMAGENES_TFG/'.
            norm (int, optional): _description_. Defaults to 50.
        """
        database = pd.read_csv(ruta+'dataset_nuevo.csv')  
        files = list(ruta+database["file"])
        self.files = files
        self.database = database
        self.norm = norm

    def __len__(self):
        return len(self.files)

    def __load_img_as_array__(self, idx):
        file = nib.load(self.files[idx])
        array= file.get_fdata()
        array[np.isnan(array)] = 0
        array = array/self.norm #Normalizacion de los valores de intensidad
        return array

    def __getitem__(self, idx):
        array = self.__load_img_as_array__(idx)
        image = torch.from_numpy(array.astype('float32')) 
        image = F.pad(input=image, pad=(0,5,0,19,0,5), mode='constant', value=0)
        image = torch.reshape(image, (1,96,128,96))     
        patno = self.database["PATNO"].to_numpy().astype('int16')[idx]
        year = self.database["YEAR"].to_numpy().astype('int16')[idx]
        #Sintomatologia:
        tremor = self.database["tremor"].to_numpy().astype('int16')[idx]
        tremor_on = self.database["tremor_on"].to_numpy().astype('int16')[idx]
        updrs_totscore_on = self.database["updrs_totscore_on"].to_numpy().astype('int16')[idx]
        updrs1_score = self.database["updrs1_score"].to_numpy().astype('int16')[idx]
        updrs2_score = self.database["updrs2_score"].to_numpy().astype('int16')[idx]
        updrs3_score = self.database["updrs3_score"].to_numpy().astype('int16')[idx]
        updrs4_score = self.database["updrs4_score"].to_numpy().astype('int16')[idx]
        ptau = self.database["ptau"].to_numpy().astype('int16')[idx]
        asyn = self.database["asyn"].to_numpy().astype('int16')[idx]
        rigidity = self.database["rigidity"].to_numpy().astype('int16')[idx]
        rigidity_on = self.database["rigidity_on"].to_numpy().astype('int16')[idx]
        nhy = self.database["NHY"].to_numpy().astype('int16')[idx]
        nhy_on = self.database["NHY_ON"].to_numpy().astype('int16')[idx]

        return image, patno, year, tremor, tremor_on, updrs_totscore_on, updrs1_score, updrs2_score, updrs3_score, updrs4_score, ptau, asyn, rigidity, rigidity_on, nhy, nhy_on


class ImageDatasetNuevo(ImageDataset):
    # Esto es una clase que hereda de ImageDataset, pero solo
    # reescribimos el _load_img_as_array para 
    def __init__(self, ruta='C:/TFG/IMAGENES_TFG/', norm=integral_norm, normkws={'method': 'median'}): 
        """_summary_
            UPDATE 16/03: pakitochus: He añadido el norm en el 
            constructor de la clase, de manera que luego podamos
            modificar (como he hecho) la clase 
        Args:
            ruta (str, optional): _description_. Defaults to 'C:/TFG/IMAGENES_TFG/'.
            norm (_type_, optional): _description_. Defaults to integral_norm.
            normkws (dict, optional): _description_. Defaults to {'method': 'median'}.
        """
        database = pd.read_csv(ruta+'dataset_novisimo.csv')  
        files = list(ruta+database["file"])
        self.files = files
        self.database = database
        assert callable(norm) # comprueba que norm sea una función
        self.norm = norm 
        self.normkws = normkws # argumentos de la función de normalización

    def __load_img_as_array__(self, idx):
        file = nib.load(self.files[idx])
        array= np.squeeze(file.get_fdata())
        array[np.isnan(array)] = 0
        array = self.norm(array, **self.normkws)
        return array

import torch
from torch.utils.data import Dataset

class HDFImageDataset(Dataset):
    def __init__(self, filename, norm=3, transform=None):
        """_summary_

        Args:
            filename (_type_): _description_
            norm (int, optional): _description_. Defaults to 3.
            transform (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.filename = filename
        self.file = h5py.File(self.filename, 'r')
        self.images_group = self.file['images']
        self.imagenames = list(self.images_group.keys())
        self.norm = norm
        self.transform = transform

    def __getitem__(self, index):
        imagename = self.imagenames[index]
        # Datos de identificación
        patno = self.images_group[imagename].attrs["PATNO"]
        year = self.images_group[imagename].attrs["YEAR"]
        # Imagen
        image_dataset = self.images_group[imagename]
        image_data = torch.from_numpy(image_dataset[()].squeeze().astype('float32')) 
        image_data = F.pad(input=image_data, pad=(0,5,0,19,0,5), mode='constant', value=0)
        try:
            image_data = F.sigmoid(torch.reshape(image_data, (1,96,128,96))/self.norm)
        except:
            print(f'{patno} - {year}') 

        if self.transform: 
            image_data = self.transform(image_data)

        #Sintomatologia:
        tremor = self.images_group[imagename].attrs["tremor"]
        tremor_on = self.images_group[imagename].attrs["tremor_on"]
        updrs_totscore_on = self.images_group[imagename].attrs["updrs_totscore_on"]
        updrs1_score = self.images_group[imagename].attrs["updrs1_score"]
        updrs2_score = self.images_group[imagename].attrs["updrs2_score"]
        updrs3_score = self.images_group[imagename].attrs["updrs3_score"]
        updrs4_score = self.images_group[imagename].attrs["updrs4_score"]
        ptau = self.images_group[imagename].attrs["ptau"]
        asyn = self.images_group[imagename].attrs["asyn"]
        rigidity = self.images_group[imagename].attrs["rigidity"]
        rigidity_on = self.images_group[imagename].attrs["rigidity_on"]
        nhy = self.images_group[imagename].attrs["NHY"]
        nhy_on = self.images_group[imagename].attrs["NHY_ON"]

        return image_data, patno, year, tremor, tremor_on, updrs_totscore_on, updrs1_score, updrs2_score, updrs3_score, updrs4_score, ptau, asyn, rigidity, rigidity_on, nhy, nhy_on


    def __len__(self):
        return len(self.imagenames)


import torch

class Affine3D(object):
    def __init__(self, angles, scale):
        self.rotation_angles = torch.tensor(angles)
        self.scale = torch.diag(torch.tensor([scale]*3, dtype=torch.float32))
        
    def __call__(self, volume):
        # init matrix:
        affine_matrix = torch.eye(4)

        # Convert degrees to radians
        angles_rad = torch.deg2rad(self.rotation_angles)
        
        # Compute the affine transformation matrix
        # First, the rotation matrix about the x-axis (elevation)
        cos_x = torch.cos(angles_rad[0])
        sin_x = torch.sin(angles_rad[0])
        cos_y = torch.cos(angles_rad[1])
        sin_y = torch.sin(angles_rad[1])
        cos_z = torch.cos(angles_rad[2])
        sin_z = torch.sin(angles_rad[2])

        # Compute the rotation matrix
        Rx = torch.tensor([[1, 0, 0],
                           [0, cos_x, -sin_x],
                           [0, sin_x, cos_x]])
        Ry = torch.tensor([[cos_y, 0, sin_y],
                           [0, 1, 0],
                           [-sin_y, 0, cos_y]])
        Rz = torch.tensor([[cos_z, -sin_z, 0],
                           [sin_z, cos_z, 0],
                           [0, 0, 1]])
        R = torch.mm(Rz, torch.mm(Rx, Ry))
        # Rx = torch.tensor([[1, 0, 0],
        #                    [0, torch.cos(elevation_rad), torch.sin(elevation_rad)],
        #                    [0, -torch.sin(elevation_rad), torch.cos(elevation_rad)]])
        # # Then, the rotation matrix about the y-axis (azimuth)
        # Ry = torch.tensor([[torch.cos(azimuth_rad), 0, -torch.sin(azimuth_rad)],
        #                    [0, 1, 0],
        #                    [torch.sin(azimuth_rad), 0, torch.cos(azimuth_rad)]])
        # rotation_matrix = torch.mm(Rx, Ry)
        # First , apply scale
        affine_matrix[:3, :3] = torch.mm(R, self.scale)
        
        # Create the 3D grid for the transformation
        grid = torch.nn.functional.affine_grid(affine_matrix[:3].unsqueeze(0), volume.unsqueeze(0).size())
        
        # Apply the affine transformation to the volume
        transformed_volume = torch.nn.functional.grid_sample(volume.unsqueeze(0), grid).squeeze(0)
        return transformed_volume


# DATALOADER:
#train_dataloader = DataLoader(ImageDataset, batch_size=32, shuffle=False)


# Ejemplo de visualización de imagen:
# datos = ImageDatasetNuevo(ruta='/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/', normkws={'method': 'gmm'})
# primera = datos[130][0]
# plt.title("Comprobación del funcionamiento del DataLoader")
# plt.imshow(primera[0, 30, :, :])

#%% Delete files from h5py
# import h5py

# # Open the HDF file in 'a' mode to allow modification
# with h5py.File('medical_images.hdf5', 'a') as f:
#     # Delete the dataset and its attributes
#     del f['images/3500_V12']

#%% TEST AFFINE3D:
# #%% 
# import torch
# from torch.utils.data import DataLoader
# from loader import HDFImageDataset
# import models
# from utils import *
# # from tensorboardX import SummaryWriter
# import torchvision

# # from train import *
# # from image_norms import integral_norm

# ruta = '/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/'
# # ruta = 'C:\TFG\IMAGENES_TFG/'
# num_epochs = 200
# modelo_elegido='genvae'
# d = 8
# lr = 3e-4 # 3e-4
# batch_size = 16
# PARAM_BETA = 10.
# PARAM_LOGSIGMA = np.log(.1)
# PARAM_NORM = 3
# PARAM_REDUCTION = 'mean'
# filename = f'Conv3DVAE_d{d}_BETA{int(PARAM_BETA)}_red{PARAM_REDUCTION}_lr{lr:.0E}_bs{batch_size}_n{num_epochs}_norm{PARAM_NORM}'

# train_dataset = HDFImageDataset('medical_images.hdf5', norm=PARAM_NORM)
# #%% 
# image_data, patno, year, tremor, tremor_on, updrs_totscore_on, updrs1_score, updrs2_score, updrs3_score, updrs4_score, ptau, asyn, rigidity, rigidity_on, nhy, nhy_on = train_dataset[0]
# params = {'angles': [[60, 0, 0], [0, 60, 0]], 'scale': [1.1, 0.9]}
# transform = Affine3D(**params)
# transformed_volume = transform(image_data)
# fig, ax = plt.subplots(ncols=3, figsize=(15,8))
# ax[0].imshow(torchvision.utils.make_grid(image_data.transpose(1,0).transpose(-1,0))[0])
# ax[1].imshow(torchvision.utils.make_grid(transformed_volume.transpose(1,0).transpose(-1,0))[0])
# params = {'angles': [0, 60, 0], 'scale': 1}
# transform = Affine3D(**params)
# transformed_volume = transform(image_data)
# ax[2].imshow(torchvision.utils.make_grid(transformed_volume.transpose(1,0).transpose(-1,0))[0])

# %%
