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
dataset = dataset.loc[dataset.APPRDX<3,]

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
imgs = dataset.groupby('APPRDX').sample(n=2)
#%%
files = glob.glob(DATA_PATH+'Repositorio_completo/*/Reconstructed_DaTSCAN/*/S*/PPMI_*.nii')
for index, imname in imgs.iterrows(): 
    patno = imname['PATNO']
    DX = imname['APPRDX']
    img = nib.load(DATA_PATH+imname['file'])
    imdata = img.get_fdata()
    imnorm = integral_norm(imdata, 'gmm')
    plt.figure()
    plt.hist(imnorm.flatten(), bins=100)
    plt.title(f'subject {patno} ({DX})')
    plt.xlim(0,5)

# %%
