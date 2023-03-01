###################################################################################################
# ANÁLISIS MASIVO DE DATOS DE LOS RESULTADOS OBTENIDOS:                                           #
#--------------------------------------------------------------------------------------------------
# Este script se divide en 2 partes:
#--------------------------------------------------------------------------------------------------
# 1. La primera parte se encarga de hacer un análisis cuantitativo de los resultados. Para ello   #
# se realizará regresión lineal y de procesos gaussianos (GPR).                                   #
#--------------------------------------------------------------------------------------------------
# 2. La segunda parte se encarga de hacer un análisis descriptivo de los resultados. Se observará #
# como afectan las variables latentes correlacionadas con una sintomatología específica a las     #
# imágenes reconstruidas del cerebro.                                                             #
#--------------------------------------------------------------------------------------------------
#%%
from loader import ImageDataset, DataLoader
import matplotlib.pyplot as plt 
from matplotlib import offsetbox
import numpy as np 
import pandas as pd
from tqdm import tqdm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor

###################################################################################################
# PRIMERA PARTE: ANÁLISIS CUANTITATIVO DE LOS RESULTADOS                                          #
#--------------------------------------------------------------------------------------------------
num_variables = 8
# modelo = 'PCA'
# modelo = 'CVAE'
modelo = 'CAE'
# #--------------------------------------------------------------------------------------------------
# Elección del directorio en función de los valores de las variables:
if num_variables == 2:
    if modelo == 'PCA':
        path = 'C:/TFG/Trabajo/Resultados/2_dimensiones_latentes/PCA/variables_latentes.csv'   
    elif modelo == 'CVAE':
        print("No se tienen datos de 2 variables latentes para este modelo")
    else:
        print("No se tienen datos de 2 variables latentes para este modelo")

elif num_variables == 3:
    if modelo == 'PCA':
        path = 'C:/TFG/Trabajo/Resultados/3_dimensiones_latentes/PCA/variables_latentes.csv'   
    elif modelo == 'CVAE':
        print("No se tienen datos de 16 variables latentes para este modelo")
    else:
        path = 'C:/TFG/Trabajo/Resultados/3_dimensiones_latentes/1000epochs/CAE/ModeloEntrenado/variables_latentes.csv'

elif num_variables == 8:
    if modelo == 'PCA':
        path = 'C:/TFG/Trabajo/Resultados/8_dimensiones_latentes/PCA/variables_latentes.csv'    
    elif modelo == 'CVAE':
        print("No se tienen datos de 8 variables latentes para este modelo")
    else:
        path = 'C:/TFG/Trabajo/Resultados/8_dimensiones_latentes/700epochs/CAE/ModeloEntrenado/variables_latentes.csv'

elif num_variables == 16:
    if modelo == 'PCA':
        path = 'C:/TFG/Trabajo/Resultados/16_dimensiones_latentes/PCA/variables_latentes.csv'
    elif modelo == 'CVAE':
       print("No se tienen datos de 16 variables latentes para este modelo")
    else:
        path = 'C:/TFG/Trabajo/Resultados/16_dimensiones_latentes/600epochs/CAE/ModeloEntrenado/variables_latentes.csv'
else:
    print("No se tienen datos para este número de variables latentes")
    
###################################################################################################
#                                          CARGA DE DATOS:
#-------------------------------------------------------------------------------------------------
df_latente = pd.read_csv(path)#, encoding="ISO-8859-1" )

dataset = ImageDataset()
img_list = []
tremor_list = []
tremor_on_list = []
updrs_totscore_on_list = []
updrs1_score_list = []
updrs2_score_list = []
updrs3_score_list = []
updrs4_score_list = []
ptau_list = []
asyn_list = []
rigidity_list = []
rigidity_on_list = []
nhy_list = []
nhy_on_list = []
print('------------------------------------------------------------------\n\t\tOBTENIENDO LOS DATOS DEL DATASET:\n------------------------------------------------------------------')
for i in tqdm(range(len(dataset))):
    img, patno, year, tremor, tremor_on, updrs_totscore_on, updrs1_score, updrs2_score, updrs3_score, updrs4_score, ptau, asyn, rigidity, rigidity_on, nhy, nhy_on = dataset[i]
    img_list.append(img.numpy())
    tremor_list.append(tremor)
    tremor_on_list.append(tremor_on)
    updrs_totscore_on_list.append(updrs_totscore_on)
    updrs1_score_list.append(updrs1_score)
    updrs2_score_list.append(updrs2_score)
    updrs3_score_list.append(updrs3_score)
    updrs4_score_list.append(updrs4_score)
    ptau_list.append(ptau)
    asyn_list.append(asyn)
    rigidity_list.append(rigidity)
    rigidity_on_list.append(rigidity_on)
    nhy_list.append(nhy)
    nhy_on_list.append(nhy_on)

df_latente["updrs_totscore_on"] = updrs_totscore_on_list 
df_latente["updrs1_score"] = updrs1_score_list 
df_latente["updrs2_score"] = updrs2_score_list 
df_latente["updrs3_score"] = updrs3_score_list 
df_latente["updrs4_score"] = updrs4_score_list 
df_latente["tremor"] = tremor_list
df_latente["tremor_on"] = tremor_on_list 
df_latente["rigidity"] = rigidity_list
df_latente["rigidity_on"] = rigidity_on_list 
df_latente["nhy"] = nhy_list 
df_latente["nhy_on"] = nhy_on_list
#df_latente["ptau"] = ptau_list 
#df_latente["asyn"] = asyn_list
#%%
###################################################################################################
#                                 REGRESION DE PROCESOS GAUSSIANOS:                               #
#-------------------------------------------------------------------------------------------------
# Cargo las variables lataentes en X y una variable de sintomatologia en y
X = df_latente.iloc[:, 1:num_variables+1].to_numpy()
y = df_latente["updrs_totscore_on"].to_numpy()
# Separo en conjunto de test y de entrenamiento
X_train, X_test = X[:-20], X[-20:]
y_train, y_test = y[:-20], y[-20:]
# Creo el modelo de regresión de procesos gaussianos
gpr = GaussianProcessRegressor()
# Entrenamiento del modelo usando las variables de entrenamiento
gpr.fit(X_train,y_train)
# Obtener las predicciones para los mismos datos
y_pred = gpr.predict(X_test)
# Representación de los resultados obtenidos
x = np.linspace(y_test.min(), y_test.max(), len(y_test))
plt.scatter(x, y_test, color='black', label='Real')
plt.plot(x, y_pred, color='blue', label='Predicho', linewidth=3)
plt.legend()
plt.xlabel('Índice')
plt.ylabel('Sintomatología')
plt.show()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE: ",mse, "\nCoeficiente de determinación: ",r2)
#%%
###################################################################################################
#                                        REGRESION LINEAL:                                        #
#-------------------------------------------------------------------------------------------------
# Cargo las variables lataentes en variables_X y las variables de sintomatologia en variables_Y
variables_X = df_latente.iloc[:, 1:num_variables+1].to_numpy()
variables_Y = df_latente.iloc[:,num_variables+3:].to_numpy()
# Cargo una sola variable de sintomatología
variables_Y = variables_Y[:, 0]
# Separo en conjunto de test y de entrenamiento
variables_X_train, variables_X_test = variables_X[:-20], variables_X[-20:]
variables_Y_train, variables_Y_test = variables_Y[:-20], variables_Y[-20:]
# Creo el modelo de regresión lineal
regr = linear_model.LinearRegression()
# Entrenamiento del modelo usando las variables de entrenamiento
regr.fit(variables_X_train, variables_Y_train)
# Predicciones realziadas con las variablaes de test
variables_Y_pred = regr.predict(variables_X_test)
# Representación de los resultados obtenidos
x = np.linspace(variables_Y_test.min(), variables_Y_test.max(), len(variables_Y_test))
plt.scatter(x, variables_Y_test, color='black', label='Real')
plt.plot(x, variables_Y_pred, color='blue', label='Predicho', linewidth=3)
plt.legend()
plt.xlabel('Índice')
plt.ylabel('Sintomatología')
plt.show()
print("Coeficientes: \n", regr.coef_)
print("Coeficiente de determinación: %.2f" % r2_score(variables_Y_test, variables_Y_pred))
###################################################################################################

######################################################################################################################################################################################################
######################################################################################################################################################################################################

#%%
###################################################################################################
# SEGUNDA PARTE: ANÁLISIS DESCRIPTIVO DE LOS RESULTADOS                                           #
#--------------------------------------------------------------------------------------------------
def plot_components(proj, variable_latente1, variable_latente2,
                    images=None, ax=None, thumb_frac=0.05, cmap='gray',):
    ax = ax or plt.gca()   
    ax.plot(proj[:, 0], proj[:, 1], '.k')

    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(proj.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap),
                                      proj[i])
            ax.add_artist(imagebox)
            ax.set_xlabel('Variable latente'+str(variable_latente1))
            ax.set_ylabel('Variable latente'+str(variable_latente2))
            
variable_latente1 = 1
variable_latente2 = 1
Z = df_latente.iloc[:, [variable_latente1, variable_latente2]].to_numpy()
X = np.squeeze(np.array(img_list))
#####################################################################################
#                       REPRESENTACIÓN DE LOS CORTES AXIALES:                       #
#------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 20))
plot_components(Z, # proyecciones en espacio latente
                variable_latente1, variable_latente2, # Para el título de los ejes
                images = X[:,::2,::2,40], 
                cmap='inferno')
#------------------------------------------------------------------------------------
#####################################################################################
#                       REPRESENTACIÓN DE LOS CORTES SAGITALES:                     #
#------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 20))
plot_components(Z, # proyecciones en espacio latente
                variable_latente1, variable_latente2, # Para el título de los ejes
                images = np.transpose(np.flip(X[:,55,::2,::2], axis=2), (0,2,1)), 
                cmap='inferno')
#------------------------------------------------------------------------------------
#####################################################################################
#                       REPRESENTACIÓN DE LOS CORTES CORONALES:                     #
#------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 20))
plot_components(Z, # proyecciones en espacio latente
                variable_latente1, variable_latente2, # Para el título de los ejes
                images = np.transpose(np.flip(X[:,::2,70,::2], axis=2), (0,2,1)), 
                cmap='inferno')
#------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------
# CÓDIGO PARA CREAR UN MAPA DE COLORES BASADO EN VIRIDIS PERO MÁS SATURADO                            #
#------------------------------------------------------------------------------------------------------
# # Definir el mapa de colores original
# viridis = plt.get_cmap('viridis')
# # Aumentar la saturación de cada color en el mapa
# saturated_colors = []
# for i in range(viridis.N):
#     rgb = viridis(i)[:3]  # obtener los componentes RGB del color
#     hsv = colorsys.rgb_to_hsv(*rgb)  # convertir a espacio de color HSV
#     hsv = (hsv[0], hsv[1] * 1.4, hsv[2])  # aumentar la saturación
#     rgb = colorsys.hsv_to_rgb(*hsv)  # convertir de nuevo a espacio de color RGB
#     saturated_colors.append(rgb)

# # Crear un nuevo mapa de colores con los colores saturados
# saturated_viridis = mpl.colors.LinearSegmentedColormap.from_list('saturated_viridis', saturated_colors)
# #-------------------------------------------------------------------------------------------------------




# %%
