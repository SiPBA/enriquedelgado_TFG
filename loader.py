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
#import matplotlib.pyplot as plt

# DATASET:
class ImageDataset(Dataset):
    def __init__(self, ruta='C:/TFG/IMAGENES_TFG/'):   
        database = pd.read_csv(ruta+'dataset_nuevo.csv')  
        files = list(ruta+database["file"])
        self.files = files
        self.database = database

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx, norm=50):
        file = nib.load(self.files[idx])
        array= file.get_fdata()
        array[np.isnan(array)] = 0
        array = array/norm #Normalizacion de los valores de intensidad
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


# DATALOADER:
#train_dataloader = DataLoader(ImageDataset, batch_size=32, shuffle=False)


# Ejemplo de visualización de imagen:
#datos = ImageDataset()
#primera = datos[130][0]
#plt.title("Comprobación del funcionamiento del DataLoader")
#plt.imshow(primera[0, :, 70, :])
