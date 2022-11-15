import numpy as np 
import nibabel as nib
import matplotlib.pyplot as plt
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# DATASET:
class ImageDataset(Dataset):
    def __init__(self, ruta='/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/Repositorio_completo/'):   
        files = glob.glob(ruta+'*/Reconstructed_DaTSCAN/*/*/Afin*.nii')    
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = nib.load(self.files[idx])
        array= file.get_fdata()
        array[np.isnan(array)] = 0
        array = array/50
        image = torch.from_numpy(array.astype('float32')) 
        image = F.pad(input=image, pad=(0,5,0,19,0,5), mode='constant', value=0)
        image = torch.reshape(image, (1,96,128,96))     
        return image, patno, year # actualizar


# DATALOADER:
# train_dataloader = DataLoader(ImageDataset, batch_size=32, shuffle=False)


# Ejemplo de visualización de imagen:
# datos = ImageDataset()
# primera = datos[0]
# plt.title("Comprobación del funcionamiento del DataLoader")
# plt.imshow(primera[0,:, :, 40])
