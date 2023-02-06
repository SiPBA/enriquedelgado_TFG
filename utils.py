import matplotlib. pyplot as plt

##########################################################################################################################
def df_latente(encoded_img, label1, label2, espacio_latente):
    ''' Función para obtener un dataframe con el valor de las variables latentes
        asociadas a los sujetos y al año de la visita.
    '''
    encoded_img = encoded_img.flatten().cpu().numpy()
    encoded_sample = {f"Variable {i}": enc for i, enc in enumerate(encoded_img)}  
    encoded_sample["Sujeto"] = label1
    encoded_sample["Año"] = label2
    espacio_latente.append(encoded_sample)

##########################################################################################################################
def representacion_latente(espacio_latente):
    ''' Función para realizar una representación gráfica de las variables latentes
        y la posición de un sujeto concreto con respecto a los demás.
    '''
    # Elección del paciente que se quiere seleccionar
    paciente = 1024
    print('------------------------------------------------------------------\n\t\tVISUALIZACIÓN DEL ESPACIO LATENTE:\n------------------------------------------------------------------')
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Posición del sujeto " + str(espacio_latente["Sujeto"][paciente]), fontsize=20, fontweight='bold', y=0.93)
    # Variable 0 vs Variable 1
    plt.subplot(2,3,1)
    plt.scatter(espacio_latente["Variable 0"][:], espacio_latente["Variable 1"][:], alpha=0.1)
    plt.scatter(espacio_latente["Variable 0"][paciente], espacio_latente["Variable 1"][paciente], c='r')
    plt.xlabel('Variable 0')
    plt.ylabel('Variable 1')
    # Variable 0 vs Variable 2
    plt.subplot(2,3,2)
    plt.scatter(espacio_latente["Variable 0"][:], espacio_latente["Variable 2"][:], alpha=0.1)
    plt.scatter(espacio_latente["Variable 0"][paciente], espacio_latente["Variable 2"][paciente], c='r')
    plt.xlabel('Variable 0')
    plt.ylabel('Variable 2')
    # Variable 0 vs Variable 3
    plt.subplot(2,3,3)
    plt.scatter(espacio_latente["Variable 0"][:], espacio_latente["Variable 3"][:], alpha=0.1)
    plt.scatter(espacio_latente["Variable 0"][paciente], espacio_latente["Variable 3"][paciente], c='r')
    plt.xlabel('Variable 0')
    plt.ylabel('Variable 3')
    # Variable 1 vs Variable 2
    plt.subplot(2,3,4)
    plt.scatter(espacio_latente["Variable 1"][:], espacio_latente["Variable 2"][:], alpha=0.1)
    plt.scatter(espacio_latente["Variable 1"][paciente], espacio_latente["Variable 2"][paciente], c='r')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    # Variable 1 vs Variable 3
    plt.subplot(2,3,5)
    plt.scatter(espacio_latente["Variable 1"][:], espacio_latente["Variable 3"][:], alpha=0.1)
    plt.scatter(espacio_latente["Variable 1"][paciente], espacio_latente["Variable 3"][paciente], c='r')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 3')
    # Variable 2 vs Variable 3
    plt.subplot(2,3,6)
    plt.scatter(espacio_latente["Variable 2"][:], espacio_latente["Variable 3"][:], alpha=0.1)
    plt.scatter(espacio_latente["Variable 2"][paciente], espacio_latente["Variable 3"][paciente], c='r')
    plt.xlabel('Variable 2')
    plt.ylabel('Variable 3')

##########################################################################################################################