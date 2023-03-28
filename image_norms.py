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

# DATA_PATH = '/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/'

# dataset = pd.read_csv(DATA_PATH+'/dataset_novisimo.csv', index_col=0)
# # dataset = dataset.loc[dataset.APPRDX<3,]
# for cols in ['EVENT_ID', 'visit_date', 'ST_startdate', 'othneuro',
#        'symptom5_comment', 'tau_txt', 'ptau_txt', 'APOE', 'SNCA_rs356181',
#        'SNCA_rs3910105', 'MAPT', 'file']:
#     dataset[cols] = dataset[cols].astype("string")

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
    if method=='gmm':
        return X/norm, model
    return X/norm


#%% 
# imgs = dataset.groupby('APPRDX').sample(n=2)
#%% CHECK REGISTRATION
# import os
# REPORT_PATH = '/home/pakitochus/Descargas/REPORT_GMM_NORM'
# from tqdm import tqdm
# exception_indices = []

def figure_report(imdata, imnorm, imname, params, REPORT_PATH, success=True): 
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(17,10))
    patno, year, DX = imname["PATNO"], imname["YEAR"], imname['APPRDX']
    filename = f'{patno}_{year}'
    if success==False:
        filename = 'ERROR_'+filename
    pcm = ax[0][0].imshow(imdata[...,40])
    ax[0][0].set_title('imagen original')
    fig.colorbar(pcm, ax=ax[0][0])
    ax[0][2].text(.0, .6, f'SUBJECT: {patno}')
    ax[0][2].text(.0, .5, f'YEAR: {year}')
    ax[0][2].text(.0, .4, f'DX: {DX}')
    ax[0][2].axis('off')
    # Histograma de la imagen original: 
    ax[1][0].hist(imdata.flatten(), bins=50)
    ax[1][0].set_title('Histograma de la imagen original')
    if success:
        pcm = ax[0][1].imshow(imnorm[...,40], vmin=0, vmax=5)
        ax[0][1].set_title('imagen normalizada')
        fig.colorbar(pcm, ax=ax[0][1])
        # Histograma de la imagen normalizada: 
        ax[1][1].hist(imnorm.flatten(), bins=50)
        ax[1][1].set_title('Histograma de la imagen Normalizada')
        # Texto de los parámetros de normalización 
        ax[1][2].text(.0, .8, 'NORM PARAMETERS')
        ax[1][2].text(.0, .7, 'MEANS')
        ax[1][2].text(.0, .6, str(params))
    # ax[1][2].text(.0, .5, 'COVARIANCES')
    # ax[1][2].text(.0, .4, str(model.covariances_.flatten()))
    # ax[1][2].text(.0, .3, 'WEIGHTS')
    # ax[1][2].text(.0, .2, str(model.weights_))
    ax[1][2].axis('off')
    fig.savefig(os.path.join(REPORT_PATH, filename+'.jpg'))
    plt.close(fig)

# for index, imname in tqdm(dataset.iterrows(), total=len(dataset)): 
#     img = nib.load(DATA_PATH+imname['file'])
#     patno, year, DX = imname["PATNO"], imname["YEAR"], imname['APPRDX']
#     filename = f'{patno}_{year}'
#     imdata = img.get_fdata().squeeze()
#     imnorm = imdata.copy()
#     # model = GaussianMixture(3, covariance_type='diag')
#     # model.fit(imnorm.reshape(-1, 1))
#     try:
#         norm = np.nanmean(imdata[20:72,20:45,20:75])
#         bg = np.nanmean(imdata[bgim==0])
#         imnorm -= bg
#         norm -= bg
#         imnorm /= norm
#         imnorm = np.clip(imnorm, 0, None)
#         params = {'norm': norm, 'bg': bg}
#         # select = model.covariances_> 2
#         # select = select & (model.weights_>.1).reshape(-1,1)
#         # mms = model.means_[select]
#         # norm = sorted(mms)[-1]
#         # imnorm -= min(model.means_)
#         # norm -= min(model.means_)
#         # imnorm /= norm
#         # imnorm = np.clip(imnorm, 0, None)
#     except Exception as e:
#         # Append the index of the current image to the exception_indices list
#         exception_indices.append(index)
#         print(e)
#         figure_report(imdata, None, imname, params, REPORT_PATH, success=False)
#         continue

# figure_report(imdata, imnorm, imname, params, REPORT_PATH, success=True)


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

# # %% SCRIPT TO GENERATE HDF FILE WITH IMAGES.
# import h5py
# import os
# from tqdm import tqdm
# REPORT_PATH = '/home/pakitochus/Descargas/REPORT_GMM_NORM'
# filename = 'medical_images.hdf5'
# exception_indices = []
# bgimfil = nib.load('/home/pakitochus/Universidad/Docencia/TFEs/EnriqueDelgado/IntensityNorm_Afin_PPMI_mask.nii')
# bgim = bgimfil.get_fdata().squeeze()

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
#         imdata = img.get_fdata().squeeze()
#         imnorm = imdata.copy()
#         try:
#             norm = np.nanmean(imdata[20:72,20:45,20:75])
#             bg = np.nanmean(imdata[bgim==0])
#             imnorm -= bg
#             norm -= bg
#             imnorm /= norm
#             imnorm = np.clip(imnorm, 0, None)
#             params = {'norm': norm, 'bg': bg}

#         except Exception as e:
#             # Append the index of the current image to the exception_indices list
#             exception_indices.append(index)
#             exception_indices.append(index)
#             print(e)
#             figure_report(imdata, None, imname, None, REPORT_PATH, success=False)
#             continue

#         figure_report(imdata, imnorm, imname, params, REPORT_PATH, success=True)

#         # Create a dataset for image data
#         imagename = '_'.join([str(imname['PATNO']), str(imname['EVENT_ID'])])
#         image_dataset = images_group.create_dataset(imagename, data=imnorm)
        
#         # Create attributes for the image dataset
#         for el, jal in imname.iteritems():
#             if el in dataset.dtypes[dataset.dtypes=='string'].index:             
#                 jal = str(jal)
#             image_dataset.attrs[el] = jal



# %%
