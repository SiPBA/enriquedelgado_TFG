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
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.file = h5py.File(self.filename, 'r')
        self.images_group = self.file['images']
        self.imagenames = list(self.images_group.keys())

    def __getitem__(self, index):
        imagename = self.imagenames[index]
        image_dataset = self.images_group[imagename]
        image_data = torch.from_numpy(image_dataset[()].astype('float32')) 
        image_data = F.pad(input=image_data, pad=(0,5,0,19,0,5), mode='constant', value=0)
        image_data = torch.reshape(image_data, (1,96,128,96))    
        patno = self.images_group[imagename].attrs["PATNO"]
        year = self.images_group[imagename].attrs["YEAR"]
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

# DATALOADER:
#train_dataloader = DataLoader(ImageDataset, batch_size=32, shuffle=False)


# Ejemplo de visualización de imagen:
# datos = ImageDatasetNuevo(ruta='/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/', normkws={'method': 'gmm'})
# primera = datos[130][0]
# plt.title("Comprobación del funcionamiento del DataLoader")
# plt.imshow(primera[0, 30, :, :])
