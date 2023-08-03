#%% STATISTICAL ANALYSIS 
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.decomposition._factor_analysis import _ortho_rotation # varimax

d = 3
bs = 16
num_epochs=200
lr = 1E-3
norm = 3
BETA = 100
# filename = f'Conv3DVAE_d{d}_BETA{int(BETA)}_lr{lr:.0E}_bs{bs}_n{num_epochs}_norm{norm}'
filename = 'Conv3DVAE_d3_BETA1_redsum_lr1E-04_bs64_n100_norm4'
espacio_latente = pd.read_csv(filename+'.csv', index_col=0)
#%% CORRELATION ANALYSIS

espacio_rotated = espacio_latente.copy()
espacio_rotated[[f'Variable {i}' for i in range(d)]] = _ortho_rotation(espacio_latente[[f'Variable {i}' for i in range(d)]].values).T
# espacio_latente = df_latente(test_dataloader, vae, modelo_elegido, device)

g = sns.clustermap(espacio_latente.drop(['Sujeto', 'Año'], axis=1).corr(), center=0, cmap="vlag",
                   dendrogram_ratio=(.1, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75)
#%%
espacio_latente.plot(x='Variable 5', y='updrs_totscore_on', kind='scatter')
# %%
sns.lmplot(data=espacio_latente, x='Variable 0', y='updrs_totscore_on', hue='nhy')

# %% RENORMALIZE TO BL STATE: 
espacio_copia = espacio_latente.copy()
fig, ax = plt.subplots(ncols=d, nrows=4, figsize=(2.5*d, 20))
for ix, el in iter(espacio_copia.groupby('Sujeto')):
    el -= el.sort_values('Año').iloc[0]
    espacio_copia.loc[el.index,:] = el
    for ib in range(4):
        for ia in range(d):
            ax[ib][ia].plot(el[f'Variable {ia}'], el[f'updrs{ib+1}_score'])

# %%
from sklearn.decomposition import PCA
model = PCA(n_components=d)
espacio_rotated = espacio_latente.copy()
espacio_rotated[[f'Variable {i}' for i in range(d)]] = model.fit_transform(espacio_latente[[f'Variable {i}' for i in range(d)]].values)

g = sns.clustermap(espacio_rotated.drop(['Sujeto', 'Año'], axis=1).corr(), center=0, cmap="vlag",
                   dendrogram_ratio=(.1, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75)
#%% 
fig, ax = plt.subplots(ncols=d, nrows=4, figsize=(2.5*d, 20))
for ib in range(4):
    for ia in range(d):
        # ax[ib][ia].plot(el[f'Variable {ia}'], el[f'updrs{ib+1}_score'])
        sns.jointplot(
            x=f'Variable {ia}', y=f'updrs{ib+1}_score',
            data=espacio_latente, kind='hex', 
            ax=ax[ib][ia],
        )
# %%
