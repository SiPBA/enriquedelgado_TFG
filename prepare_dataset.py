#%% 
import pandas as pd 
import os
import glob 

DATA_PATH = '/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/'

dataset = pd.read_csv(DATA_PATH+'/PPMI_Original_Cohort_BL_to_Year_5_Dataset_Apr2020.csv')#, index_col=0)
dataset.visit_date = pd.to_datetime(dataset.visit_date)
dataset.ST_startdate = pd.to_datetime(dataset.ST_startdate)
dataset['file'] = pd.NA
#%%
for patno in dataset.PATNO:
    datos = dataset.loc[dataset.PATNO==patno,]
    files = glob.glob(f'{DATA_PATH}/Repositorio_completo/{patno}/Reconstructed_DaTSCAN/*/*/PPMI*.nii')
    for fil in files:
        intervals = datos.visit_date-pd.to_datetime(fil.split(os.sep)[-3].split('_')[0])
        datos.loc[intervals.abs().idxmin(),'file'] = fil.replace(DATA_PATH, '')
    dataset.loc[datos.index,'file'] = datos.file
#%%
filtered_datas = dataset.dropna(axis='rows', subset=['file'])
filtered_datas.to_csv(os.path.join(DATA_PATH, 'dataset_novisimo.csv'))
# %%
