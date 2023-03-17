#%% Imports
import numpy as np 
import nibabel as nib 
import glob
import matplotlib.pyplot as plt 
from scipy.stats import mode
from skimage import filters
from sklearn.mixture import GaussianMixture
import pandas as pd 
import os

DATA_PATH = '/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/'

dataset = pd.read_csv(DATA_PATH+'/dataset_novisimo.csv', index_col=0)
# dataset = dataset.loc[dataset.APPRDX<3,]
for cols in ['EVENT_ID', 'visit_date', 'ST_startdate', 'othneuro',
       'symptom5_comment', 'tau_txt', 'ptau_txt', 'APOE', 'SNCA_rs356181',
       'SNCA_rs3910105', 'MAPT', 'file']:
    dataset[cols] = dataset[cols].astype("string")

# imgs = ['S215424', 'S187793', 'S142202', 'S124288']

def integral_norm(X, method='median'):
    assert method in ['median', 'mean', 'mode', 'otsu', 'gmm']
    if method=='median':
        norm = np.nanmedian(X[X>0])
    elif method=='mean':
        norm = np.nanmean(X[X>0])
    elif method=='mode':
        norm = mode(X[X>0].flatten()).mode
    elif method=='otsu':
        norm = filters.threshold_otsu(X)
    elif method=='gmm':
        model = GaussianMixture(3, covariance_type='diag')
        model.fit(X.reshape(-1, 1))
        select = model.covariances_> 2
        select = select & (model.weights_>.1).reshape(-1,1)
        mms = model.means_[select]
        norm = sorted(mms)[-1]
        X -= min(model.means_)
        X = np.clip(X, 0, None)
    return X/norm


#%% 
# imgs = dataset.groupby('APPRDX').sample(n=2)
#%%
# files = glob.glob(DATA_PATH+'Repositorio_completo/*/Reconstructed_DaTSCAN/*/S*/PPMI_*.nii')
# for index, imname in imgs.iterrows(): 
#     patno = imname['PATNO']
#     DX = imname['APPRDX']
#     img = nib.load(DATA_PATH+imname['file'])
#     imdata = img.get_fdata()
#     imnorm = integral_norm(imdata, 'gmm')
#     plt.figure()
#     plt.hist(imnorm.flatten(), bins=100)
#     plt.title(f'subject {patno} ({DX})')
#     plt.xlim(0,5)

# # %%
# import h5py
# import os
# from tqdm import tqdm

# filename = 'medical_images.hdf5'
# exception_indices = []

# if os.path.exists(filename):
#     os.remove(filename)

# # Creamos un archivo HDF
# with h5py.File(filename, 'w') as f:
#     pass

# # Creamos un grupo de imágenes,d entro del archivo: 
# with h5py.File(filename, 'a') as f:
#     images_group = f.create_group('images')

# # Iteramos por todas las imágenes disponibles y añadimos: 
# for index, imname in tqdm(dataset.iterrows(), total=len(dataset)): 
#     # if index>=2079: # se me quedó colgado aquí. 
#     with h5py.File(filename, 'a') as f:
#         images_group = f['images'] 

#         img = nib.load(DATA_PATH+imname['file'])
#         imdata = img.get_fdata()
#         try:
#             imnorm = integral_norm(imdata, 'gmm')
#         except Exception as e:
#             # Append the index of the current image to the exception_indices list
#             exception_indices.append(index)
#             continue

#         # Create a dataset for image data
#         imagename = '_'.join([str(imname['PATNO']), str(imname['EVENT_ID'])])
#         image_dataset = images_group.create_dataset(imagename, data=imnorm)
        
#         # Create attributes for the image dataset
#         for el, jal in imname.iteritems():
#             if el in dataset.dtypes[dataset.dtypes=='string'].index:             
#                 jal = str(jal)
#             image_dataset.attrs[el] = jal



# %%
