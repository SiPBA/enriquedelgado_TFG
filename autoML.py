#%% 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
from supervised.automl import AutoML # mljar-supervised
import glob as glob
import os
#%% 


MODE = 'Explain'
filenames = glob.glob('*.csv')
targets = ['tremor_on', 'updrs_totscore_on', 'updrs1_score', 'updrs2_score',
       'updrs3_score', 'updrs4_score', 'ptau', 'asyn', 'rigidity_on']
for filename in filenames:
    filename = filename.replace('.csv', '')
    d = int(filename.split('_')[1].replace('d', ''))
    # filename = 'Conv3DVAE_d8_BETA100_lr1E-03_bs16_n200_norm3'
    # filename = 'Conv3DVAE_d20_BETA100_lr1E-03_bs16_n200_norm3'
    # filename = 'Conv3DVAE_d3_BETA1_redsum_lr1E-04_bs64_n100_norm4'
    espacio_latente = pd.read_csv(filename+'.csv', index_col=0)
    espacio_latente.columns = [el.replace(' ', '') for el in espacio_latente.columns]

    for target in targets:
        results_path='AutoML_results/'+MODE+'/'+target+'/'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        results_path += filename

        matrix_target = espacio_latente.dropna(axis=0, subset=target)
        # Load the data
        X_train, X_test, y_train, y_test = train_test_split(
            matrix_target[[f'Variable{i}' for i in range(d)]],
            matrix_target[target],
            test_size=0.25,
            random_state=123,
        )

        # train models with AutoML
        automl = AutoML(mode=MODE, ml_task='regression', 
                        results_path=results_path, features_selection=True, 
                        explain_level=2, golden_features=True, kmeans_features=True,
                        n_jobs=2)
        automl.fit(X_train, y_train)

        # compute the MSE on test data
        predictions = automl.predict(X_test)
        print("Test MSE:", mean_squared_error(y_test, predictions))
# %%

MODE = 'Perform'
filenames = glob.glob('*.csv')
targets = ['nhy_on']
for filename in filenames:
    filename = filename.replace('.csv', '')
    d = int(filename.split('_')[1].replace('d', ''))
    # filename = 'Conv3DVAE_d8_BETA100_lr1E-03_bs16_n200_norm3'
    # filename = 'Conv3DVAE_d20_BETA100_lr1E-03_bs16_n200_norm3'
    # filename = 'Conv3DVAE_d3_BETA1_redsum_lr1E-04_bs64_n100_norm4'
    espacio_latente = pd.read_csv(filename+'.csv', index_col=0)
    espacio_latente.columns = [el.replace(' ', '') for el in espacio_latente.columns]

    
    for target in targets:
        results_path='AutoML_results/'+MODE+'/'+target+'/'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        results_path += filename
        # Load the data
        X_train, X_test, y_train, y_test = train_test_split(
            espacio_latente[[f'Variable{i}' for i in range(d)]],
            espacio_latente[target],
            test_size=0.25,
            random_state=123,
        )

        # train models with AutoML
        automl = AutoML(mode=MODE, ml_task='multiclass_classification', 
                        results_path=results_path, features_selection=True, 
                        explain_level=2, golden_features=True, kmeans_features=True,
                        n_jobs=5)
        automl.fit(X_train, y_train)

        # compute the MSE on test data
        predictions = automl.predict(X_test)
        print("Test BALACC:", balanced_accuracy_score(y_test, predictions))

# %% GENERATES PERFORMANCE METRICS
import glob
import re
import pandas as pd

# Define the table shape pattern
table_shape = r"\|\s*(\w*)\s*\|\s* (-?\d+\.\d+)?([eE][-+]?\d+)?\s*\|"
name_shape = r"([a-zA-Z]+)(\d+)?([eE][-+]?\d+)?"

def findall(pattern, string):
    while True:
        match = re.search(pattern, string)
        if not match:
            break
        if match.group(3):
            exponent = match.group(3)
        else:
            exponent = ''
        yield [match.group(1), float(match.group(2)+exponent)]
        string = string[match.end():]

# Create an empty DataFrame to store the table data
df = pd.DataFrame()

# Iterate over folders
for i, folder in enumerate(glob.glob("/home/pakitochus/Universidad/Docencia/TFEs/EnriqueDelgado/enriquedelgado_TFG/AutoML_results/Explain/*/*/*/")):
    # Load README.md file
    readme_file = folder + "/README.md"
    try:
        with open(readme_file, "r") as f:
            readme_content = f.read()
    except FileNotFoundError:
        print(f"README.md file not found in folder: {folder}")
        continue

    aux = pd.DataFrame([folder.split('/')[9:-1]],
                       index=[i], 
                       columns=['mode', 'target', 'string', 'model'])
    for key, value in findall(table_shape, readme_content):
        if ('Variable' in key) or ('Dist' in key):
            continue
        aux[key] = value

    strname = aux['string'].item()
    aux['latent_model'] = strname.split('_')[0]
    if 'redmean' in strname:
        aux['reduction'] = 'redmean'
        strname = strname.replace('redmean', '')
    if 'redsum' in strname:
        aux['reduction'] = 'redsum'
        strname = strname.replace('redsum', '')
    strname = strname.replace(aux['latent_model'].item(), '')
    for key, value in findall(name_shape, strname):
        aux[key] = value


    # Convert dictionary to a DataFrame and append to the main DataFrame
    df = pd.concat((df, aux), axis=0)

# Display the resulting DataFrame
print(df)
df = df.loc[df.model!='Ensemble',]
#%% 
df_updrs = df.loc[df.target.str.contains('updrs'), ]
df_noupdrs = df.loc[~df.target.str.contains('updrs'), ]
df_filtered = df_updrs.loc[df_updrs.model.str.contains('Xgboost')|df_updrs.model.str.contains('DecisionTree'),]
# df_filtered = df_filtered.loc[~df_filtered.model.str.contains('GoldenFeatures'),]
# df_filtered = df_filtered.loc[~df_filtered.model.str.contains('RandomFeature'),]
# df_filtered = df_filtered.loc[~df_filtered.model.str.contains('SelectedFeatures'),]
df_filtered = df_filtered.loc[df_filtered.R2>-1,]
# %%
idxs = df_filtered.groupby(['target', 'd']).idxmax()['R2']
print(df_filtered.loc[idxs, ['target', 'd', 'model', 'MAE', 'RMSE', 'R2']].T.to_latex(float_format='%.3f'))
# %%
import matplotlib.pyplot as plt 
import seaborn as sns
df_filtered['target'] = df_filtered.target.str.replace('_score', '')
df_filtered['target'] = df_filtered.target.str.replace('updrs_totscore_on', 'updrs (total)')
df_filtered['d'] = df_filtered['d'].astype(int)
fig, ax = plt.subplots(figsize=(6,3))
sns.swarmplot(x='target', y='R2', hue='d', data=df_filtered, palette='muted', ax=ax, order=['updrs1', 'updrs2', 'updrs3', 'updrs4', 'updrs (total)'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_title('R2 for each target and latent space dimension')
# %%
