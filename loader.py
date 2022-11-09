import numpy as np 
import nibabel as nib
import matplotlib.pyplot as plt
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

files = glob.glob('C:\TFG\IMAGENES_TFG/Repositorio_completo/*/Reconstructed_DaTSCAN/*/*/Afin*.nii')

# DATASET:
class ImageDataset(Dataset):
    def __init__(self):       
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = nib.load(files[idx])
        array= file.get_fdata()
        image = torch.from_numpy(array.astype('float32')) 
        image = F.pad(input=image, pad=(0,5,0,19,0,5), mode='constant', value=0)
        image = torch.reshape(image, (1,96,128,96))     
        return image


# DATALOADER:
train_dataloader = DataLoader(ImageDataset, batch_size=32, shuffle=False)


# Ejemplo de visualización de imagen:
datos = ImageDataset()
primera = datos[0]
plt.title("Comprobación del funcionamiento del DataLoader")
plt.imshow(primera[0,:, :, 40])
