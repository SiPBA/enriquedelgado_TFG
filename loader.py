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
    def __init__(self, ruta='/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/'):   
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
        patno = self.database["PATNO"].to_numpy().astype('int16')
        year = self.database["YEAR"].to_numpy().astype('int16')
        return image, patno, year 


# DATALOADER:
# train_dataloader = DataLoader(ImageDataset, batch_size=32, shuffle=False)


# Ejemplo de visualización de imagen:
# datos = ImageDataset()
# primera = datos[0]
# plt.title("Comprobación del funcionamiento del DataLoader")
# plt.imshow(primera[0,:, :, 40])
