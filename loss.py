###################################################################################################
# DEFINICIÓN DE LAS FUNCIONES DE PÉRDIDAS DE LOS MODELOS:                                         #
#--------------------------------------------------------------------------------------------------
# Este script define 2 funciones de pérdidas para los modelos:                                    #
#--------------------------------------------------------------------------------------------------
# 1. loss_CAE: Función de pérdidas para el AutoEncoder Convolucional. Realiza el error cuadrático #
#              medio.                                                                             #
#--------------------------------------------------------------------------------------------------
# 2. loss_function: Función de pérdidas para el AutoEncoder Convolucional Variacional. Realiza el #
#                   cómputo de la función de pérdidas sumando el error de reconstrucción y el     #
#                   producto del parámetro beta por la divergencia de kullback-leibler.           #
#--------------------------------------------------------------------------------------------------
import torch

# Para el CAE:
def loss_CAE():
    ''' Función que realiza el cómputo de la función de pérdidas por mínimos cuadrados'''
    loss_fn = torch.nn.MSELoss()
    return loss_fn

#--------------------------------------------------------------------------------------------------
# Para el CVAE:
def loss_function(image_batch,decoded_data, mu_x, logvar_x, mu_latent, logvar_latent, beta=1.):
    '''Función que realiza el cómputo de la función de pérdidas bajo una asunción de gaussianidad. '''
    # neg log likelihood of x under normal assumption
    LOG_2_PI = torch.log(2.0 * torch.acos(torch.zeros(1))).item()
    loss_rec = -torch.sum((-0.5 * LOG_2_PI + (-0.5 * logvar_x) + (-0.5 / torch.exp(logvar_x)) * (image_batch - mu_x) ** 2.0), dim=[1,2,3,4])
    #------------------------------------------------------------------------------------- 
    KLD = -0.5 * torch.sum(1 + logvar_latent- mu_latent.pow(2) - logvar_latent.exp(), dim=1)
    return torch.mean(loss_rec + beta * KLD)

#--------------------------------------------------------------------------------------------------
