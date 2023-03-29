###################################################################################################
# DEFINICIÓN DE FUNCIONES ÚTILES PARA EL SISTEMA:                                                 #
#--------------------------------------------------------------------------------------------------
# Este script contiene 6 funciones de utilidad para el sistema completo:                          #
#--------------------------------------------------------------------------------------------------
# 1. df_latente: Obtiene los valores del espacio latente y los almacena en un dataframe           #
#--------------------------------------------------------------------------------------------------
# 2. representa_latente: Realiza un subplot con varias representaciones de variables latentes.    #
#--------------------------------------------------------------------------------------------------
# 3. guarda_imag: Guarda 3 cortes de la imagen reconstruida por el decoder en cada epoch.         #
#--------------------------------------------------------------------------------------------------
# 4. anima_latente: Guarda 3 gráficas con distintas representaciones de variables latentes frente #
#                   a otras.                                                                      #
#--------------------------------------------------------------------------------------------------
# 5. representa_perdidas: Representa la función de pérdidas del entrenamiento del modelo.         #
#--------------------------------------------------------------------------------------------------
# 6. guarda_perdidas: Guarda la representación de la función de pérdidas en cada epoch.           #
#--------------------------------------------------------------------------------------------------
import matplotlib. pyplot as plt
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import os

# SAVE_PATH = 'C:/TFG/Trabajo/Resultados/'
SAVE_PATH = '/home/pakitochus/Descargas/TFG-enrique/Resultados/'
#--------------------------------------------------------------------------------------------------
def df_latente(test_dataloader, model, modelo_elegido, device):
    ''' Función para obtener un dataframe con el valor de las variables latentes
        asociadas a los sujetos y al año de la visita.'''
    espacio_latente = []
    sujeto = []
    year = []
    for sample in tqdm(test_dataloader):
        img, label1, label2, tremor, tremor_on, updrs_totscore_on, updrs1_score, updrs2_score, updrs3_score, updrs4_score, ptau, asyn, rigidity, rigidity_on, nhy, nhy_on = sample
        img = img.to(device)
        label1 = str((label1.numpy()).astype("int16"))
        sujeto.append(label1)
        label2 = str((label2.numpy()).astype("int16"))
        year.append(label2)
        model.eval()

        if modelo_elegido in ['CVAE', 'genvae']:
            with torch.no_grad():
                zsample , encoded_img, _  = model.encode(img)
            encoded_img = encoded_img.flatten().cpu().numpy()      
            encoded_sample = {f"Variable {i}": enc for i, enc in enumerate(encoded_img)}  
            year = [year.strip("[]") for year in year]
            sujeto = [sujeto.strip("[]") for sujeto in sujeto]
            encoded_sample["Sujeto"] = int(sujeto[-1])
            encoded_sample["Año"] = int(year[-1])
            encoded_sample['tremor'] = int(tremor.numpy().astype('int16'))
            encoded_sample['tremor_on'] = int(tremor_on.numpy().astype('int16'))
            encoded_sample['updrs_totscore_on'] = int(updrs_totscore_on.numpy().astype('int16'))
            encoded_sample['updrs1_score'] = int(updrs1_score.numpy().astype('int16'))
            encoded_sample['updrs2_score'] = int(updrs2_score.numpy().astype('int16'))
            encoded_sample['updrs3_score'] = int(updrs3_score.numpy().astype('int16'))
            encoded_sample['updrs4_score'] = int(updrs4_score.numpy().astype('int16'))
            encoded_sample['ptau'] = float(ptau.numpy())
            encoded_sample['asyn'] = float(asyn.numpy())
            encoded_sample['rigidity'] = int(rigidity.numpy().astype('int16'))
            encoded_sample['rigidity_on'] = int(rigidity_on.numpy().astype('int16'))
            encoded_sample['nhy'] = int(nhy.numpy().astype('int16'))
            encoded_sample['nhy_on'] = int(nhy_on.numpy().astype('int16'))
            espacio_latente.append(encoded_sample)

        elif modelo_elegido == 'CAE':
            with torch.no_grad():
                encoded_img  = model.encode(img)
            encoded_img = encoded_img.flatten().cpu().numpy()      
            encoded_sample = {f"Variable {i}": enc for i, enc in enumerate(encoded_img)}  
            year = [year.strip("[]") for year in year]
            sujeto = [sujeto.strip("[]") for sujeto in sujeto]
            encoded_sample["Sujeto"] = int(sujeto[-1])
            encoded_sample["Año"] = int(year[-1])
            espacio_latente.append(encoded_sample)

        else:
            print('Modelo no definido')

    espacio_latente = pd.DataFrame(espacio_latente)
    return espacio_latente

#--------------------------------------------------------------------------------------------------
def representacion_latente(espacio_latente):
    ''' Función para realizar una representación gráfica de las variables latentes
        y la posición de un sujeto concreto con respecto a los demás.'''
    # Elección del paciente que se quiere seleccionar
    paciente = 1120
    print('------------------------------------------------------------------\n\t\tVISUALIZACIÓN DEL ESPACIO LATENTE:\n------------------------------------------------------------------')
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Posición del sujeto " + str(espacio_latente["Sujeto"][paciente]), fontsize=20, fontweight='bold', y=0.93)
    # Variable 0 vs Variable 1
    plt.subplot(1,2,1)
    plt.scatter(espacio_latente["Variable 0"][:], espacio_latente["Variable 1"][:], color=[0.5647, 0.0471, 0.2471], alpha=0.1)
    plt.scatter(espacio_latente["Variable 0"][paciente], espacio_latente["Variable 1"][paciente], color=[0,0.3,1])
    plt.xlabel('Variable 0')
    plt.ylabel('Variable 1')
    # Variable 0 vs Variable 2
    # plt.subplot(1,2,2)
    # plt.scatter(espacio_latente["Variable 0"][:], espacio_latente["Variable 2"][:], color=[0.5647, 0.0471, 0.2471], alpha=0.1)
    # plt.scatter(espacio_latente["Variable 0"][paciente], espacio_latente["Variable 2"][paciente], color=[0,0.3,1])
    # plt.xlabel('Variable 0')
    # plt.ylabel('Variable 2')
    # Variable 0 vs Variable 3
    # plt.subplot(2,3,3)
    # plt.scatter(espacio_latente["Variable 0"][:], espacio_latente["Variable 3"][:], color=[0.5647, 0.0471, 0.2471], alpha=0.1)
    # plt.scatter(espacio_latente["Variable 0"][paciente], espacio_latente["Variable 3"][paciente], color=[0,0.3,1])
    # plt.xlabel('Variable 0')
    # plt.ylabel('Variable 3')
    # # Variable 1 vs Variable 2
    # plt.subplot(2,3,4)
    # plt.scatter(espacio_latente["Variable 1"][:], espacio_latente["Variable 2"][:], color=[0.5647, 0.0471, 0.2471], alpha=0.1)
    # plt.scatter(espacio_latente["Variable 1"][paciente], espacio_latente["Variable 2"][paciente], color=[0,0.3,1])
    # plt.xlabel('Variable 1')
    # plt.ylabel('Variable 2')
    # # Variable 1 vs Variable 3
    # plt.subplot(2,3,5)
    # plt.scatter(espacio_latente["Variable 1"][:], espacio_latente["Variable 3"][:], color=[0.5647, 0.0471, 0.2471], alpha=0.1)
    # plt.scatter(espacio_latente["Variable 1"][paciente], espacio_latente["Variable 3"][paciente], color=[0,0.3,1])
    # plt.xlabel('Variable 1')
    # plt.ylabel('Variable 3')
    # Variable 2 vs Variable 3
    # plt.subplot(2,3,6)
    # plt.scatter(espacio_latente["Variable 2"][:], espacio_latente["Variable 3"][:], color=[0.5647, 0.0471, 0.2471], alpha=0.1)
    # plt.scatter(espacio_latente["Variable 2"][paciente], espacio_latente["Variable 3"][paciente], color=[0,0.3,1])
    # plt.xlabel('Variable 2')
    # plt.ylabel('Variable 3')

#--------------------------------------------------------------------------------------------------
def guarda_imag(decoded_data, patno, year, epoch, modelo_elegido, num_epochs, d):
    ''' Función para realizar una representación gráfica de las imágenes
        reconstruidas por el decoder a partir de las variables latentes y guardarlas.'''
    ###############################################################################################################
    #                           CÓDIGO PARA GUARDAR LAS IMÁGENES RECONSTRUIDAS POR EL DECODER:                    #
    #--------------------------------------------------------------------------------------------------------------
    # Para guardarla en 3D (ESTE CÓDIGO ESTÁ SIN TERMINAR POR FALTA DE UTILIDAD):
    #--------------------------------------------------------------------------------------------------------------
    #path = 'C:/TFG/Resultados'+str(num_epoch)+'epochs/'+modelo_elegido+'/ImagenesReconstruidas/3D/decoded' + str(image_no)+'.nii'
    #nib.save(nib.Nifti1Image(decoded_data.cpu().detach().numpy()[0,0,:,:,:].astype('float64'), np.eye(4)), path)
    #image_no += 1
    #--------------------------------------------------------------------------------------------------------------
    # Para guardar varias secciones en 2D:
    #--------------------------------------------------------------------------------------------------------------
    #aux = 30 # Variable para la generación de imagenes 2D. Indica los FPS a los que se va a animar
    for idx in range(1):  
        # ELEGIR DIRECTORIO:
        dirname = os.path.join(SAVE_PATH, str(d)+'_dimensiones_latentes', str(num_epochs)+'epochs', modelo_elegido, 'ImagenesReconstruidas')
        patname = 'Paciente_'+ str(patno.numpy()[idx])+'_'+str(year.numpy()[idx])+'_'+str(epoch)+'.jpg'         
        #CORTE AXIAL:
        plt.imshow(decoded_data.cpu().detach().numpy()[0,0,0:91,0:108,40], cmap='inferno')
        plt.title("Reconstrucción vista desde un\n corte axial del cerebro", dict(size=15))
        #plt.text(105, -5, str(aux)+'FPS', dict(size=20), color='red') 
        if not os.path.exists(os.path.join(dirname, 'Axial')):
            os.makedirs(os.path.join(dirname, 'Axial'))
        plt.savefig(os.path.join(dirname, 'Axial', patname))
        plt.clf()
        # CORTE SAGITAL:
        plt.imshow(np.fliplr(np.flip(np.transpose(decoded_data.cpu().detach().numpy()[0,0,55,0:108,0:91]))), cmap='inferno')
        plt.title("Reconstrucción vista desde un\n corte sagital del cerebro", dict(size=15))
        # ELEGIR DIRECTORIO:
        if not os.path.exists(os.path.join(dirname, 'Sagital')):
            os.makedirs(os.path.join(dirname, 'Sagital'))
        plt.savefig(os.path.join(dirname, 'Sagital', patname))
        plt.clf()
        # CORTE CORONAL:
        plt.imshow(np.flip(np.transpose(decoded_data.cpu().detach().numpy()[0,0,0:91,70,0:91])), cmap='inferno')
        plt.title("Reconstrucción vista desde un\n corte coronal del cerebro", dict(size=15))
        # ELEGIR DIRECTORIO:
        if not os.path.exists(os.path.join(dirname, 'Coronal')):
            os.makedirs(os.path.join(dirname, 'Coronal'))
        plt.savefig(os.path.join(dirname, 'Coronal', patname))
        plt.clf()

#--------------------------------------------------------------------------------------------------
def anima_latente(espacio_latente, epoch, modelo_elegido, num_epoch, d):
    '''Función para representar y guardar el espacio latente de sujetos concretos
       en cada epoch del entrenamiento para luego realizar una animación.'''
    # Elección del juento que se quiere visualizar (Los sujetos correspondientes a los índices
    # 704 y 1120 son en este caso el 3540_4 y el 3853_0 respectivamente con updrs_totscore_on de 1140 y 10 respectivamente)
    sujeto = [704, 1120]
    for idx in range(2):
        # ELEGIR DIRECTORIO:
        paciente = sujeto[idx]
        dirname = os.path.join(SAVE_PATH, str(d)+'_dimensiones_latentes', str(num_epoch)+'epochs', modelo_elegido, 'ImagenesReconstruidas')
        patname = 'Paciente_'+ str(espacio_latente["Sujeto"][paciente])+'_'+str(espacio_latente["Año"][paciente])+"_"+str(epoch)+'.jpg'      

        # Variable 0 vs Variable 1
        plt.scatter(espacio_latente["Variable 0"][:], espacio_latente["Variable 1"][:], color=[0.5647, 0.0471, 0.2471], alpha=0.1)
        plt.scatter(espacio_latente["Variable 0"][paciente], espacio_latente["Variable 1"][paciente], color=[0,0.3,1])
        plt.xlabel('Variable 0')
        plt.ylabel('Variable 1')
        # plt.xlim([-3000, 2200])
        # plt.ylim([-3500, 2400])
        if modelo_elegido == 'CAE':
            plt.xlim([-1500, 1500])
            plt.ylim([-1500, 1500])
        else:
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
        plt.title("Evolución del sujeto " + str(espacio_latente["Sujeto"][paciente]))
        if not os.path.exists(os.path.join(dirname, 'EspacioLatente1')):
            os.makedirs(os.path.join(dirname, 'EspacioLatente1'))
        plt.savefig(os.path.join(dirname, 'EspacioLatente1', patname))
        plt.clf()
        # # Variable 2 vs Variable 3
        # plt.scatter(espacio_latente["Variable 1"][:], espacio_latente["Variable 2"][:], color=[0.5647, 0.0471, 0.2471], alpha=0.1)
        # plt.scatter(espacio_latente["Variable 1"][paciente], espacio_latente["Variable 2"][paciente], color=[0,0.3,1])
        # plt.xlabel('Variable 2')
        # plt.ylabel('Variable 3')
        # # plt.xlim([-3000, 2200])
        # # plt.ylim([-2500, 2400])
        # if modelo_elegido == 'CAE':
        #     plt.xlim([-1500, 1500])
        #     plt.ylim([-1500, 1500])
        # else:
        #     plt.xlim([-10, 10])
        #     plt.ylim([-10, 10])
        # plt.title("Evolución del sujeto " + str(espacio_latente["Sujeto"][paciente]))
        # # ELEGIR DIRECTORIO:
        # if not os.path.exists(os.path.join(dirname, 'EspacioLatente2')):
        #     os.makedirs(os.path.join(dirname, 'EspacioLatente2'))
        # plt.savefig(os.path.join(dirname, 'EspacioLatente2', patname))        
        # plt.clf()
        # # Variable 4 vs Variable 5
        # plt.scatter(espacio_latente["Variable 0"][:], espacio_latente["Variable 2"][:], color=[0.5647, 0.0471, 0.2471], alpha=0.1)
        # plt.scatter(espacio_latente["Variable 0"][paciente], espacio_latente["Variable 2"][paciente], color=[0,0.3,1])
        # plt.xlabel('Variable 4')
        # plt.ylabel('Variable 5')
        # # plt.xlim([-3000, 2200])
        # # plt.ylim([-2500, 2400])
        # if modelo_elegido == 'CAE':
        #     plt.xlim([-1500, 1500])
        #     plt.ylim([-1500, 1500])
        # else:
        #     plt.xlim([-10, 10])
        #     plt.ylim([-10, 10])
        # plt.title("Evolución del sujeto " + str(espacio_latente["Sujeto"][paciente]))
        # # ELEGIR DIRECTORIO:
        # if not os.path.exists(os.path.join(dirname, 'EspacioLatente3')):
        #     os.makedirs(os.path.join(dirname, 'EspacioLatente3'))
        # plt.savefig(os.path.join(dirname, 'EspacioLatente3', patname))  
        # plt.clf()

#--------------------------------------------------------------------------------------------------
def representa_perdidas(diz_loss, modelo_elegido):
    '''Función que realiza la representación de la función
       de pérdidas después del entrenamiento.'''
    if modelo_elegido == 'CVAE':
        plt.figure(figsize=(10,8))
        plt.plot(diz_loss['train_loss'], label='Train')
        plt.xlabel('Epoch')
        plt.ylabel('Pérdidas')
        plt.grid()
        plt.legend()
        plt.title('Función de pérdidas')
        plt.show()
        plt.clf()
    else:    
        plt.figure(figsize=(10,8))
        plt.semilogy(diz_loss['train_loss'], label='Train')
        plt.xlabel('Epoch')
        plt.ylabel('Pérdidas')
        plt.grid()
        plt.legend()
        plt.title('Función de pérdidas')
        plt.show()
        plt.clf()

#--------------------------------------------------------------------------------------------------
def guarda_perdidas(diz_loss, modelo_elegido, num_epoch, d):
    '''Función que guarda segmentada la función de pérdidas para 
        realizar luego la animación.'''
    if modelo_elegido == 'CVAE':
        for i in tqdm(range(num_epoch)):
            plt.figure(figsize=(6.4,4.8)) # Para que coincidan las dimensiones para la animación.
            plt.plot(diz_loss['train_loss'][0:i], c='r', label='Train')
            plt.xlabel('Epoch')
            plt.ylabel('Pérdidas')
            plt.grid()
            plt.xlim([-5, num_epoch])
            plt.ylim([-4e6, 0.6e6])
            plt.legend()
            plt.title('Función de pérdidas')
            plt.savefig(SAVE_PATH+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo_elegido+'/FuncionPerdidas/decoded'+ str(i)+'.jpg')
            plt.clf()
            plt.close()
    else:
        for i in tqdm(range(num_epoch)):
            plt.figure(figsize=(6.4,4.8)) # Para que coincidan las dimensiones para la animación.
            plt.semilogy(diz_loss['train_loss'][0:i], c='r', label='Train')
            plt.xlabel('Epoch')
            plt.ylabel('Pérdidas')
            plt.grid()
            plt.xlim([-10, num_epoch])
            plt.ylim([4e-3, 0.3])
            plt.legend()
            plt.title('Función de pérdidas')
            plt.savefig(SAVE_PATH+str(d)+'_dimensiones_latentes/'+str(num_epoch)+'epochs/'+modelo_elegido+'/FuncionPerdidas/decoded'+ str(i)+'.jpg')
            plt.clf()
            plt.close()

#--------------------------------------------------------------------------------------------------