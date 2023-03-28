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
import torch.nn.functional as F

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
    #------------------------------------------------------------------------------------- 
    # logvar_latent = torch.clip(torch.exp(logvar_latent), min=1e-5) # ayuda a que logvar no sea 0 y evita nans. 
    loss_rec = F.mse_loss(mu_x, image_batch, reduction='sum')
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp(), dim=1), dim = 0)
    return torch.mean(loss_rec + beta * KLD)

#--------------------------------------------------------------------------------------------------
def gaussian_likelihood(x_hat, logscale, x):
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3, 4))

def lognormal_likelihood(x_hat, logscale, x):
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.LogNormal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.mean(dim=(1, 2, 3, 4))

def kl_divergence(z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl

# CVAE alternativas:
def loss_vae_gauss(image_batch,decoded_data, z_sample, logvar_x, mu_latent, logvar_latent, beta=1.):
    '''Función que realiza el cómputo de la función de pérdidas bajo una asunción de gaussianidad. '''
    # logvar_x = torch.clip(torch.exp(logvar_x), min=1e-5) # ayuda a que logvar no sea 0 y evita nans. 
    # neg log likelihood of x under normal assumption
    # LOG_2_PI = torch.log(2.0 * torch.acos(torch.zeros(1))).item()
    # loss_rec = -torch.sum((-0.5 * LOG_2_PI + (-0.5 * logvar_x) + (-0.5 / torch.exp(logvar_x)) * (image_batch - mu_x) ** 2.0), dim=[1,2,3,4])
    loss_rec = gaussian_likelihood(decoded_data, logvar_x, image_batch)
    #--------------------------------------axz<><----------------------------------------------- 
    # KLD = -0.5 * torch.sum(1 + logvar_latent- mu_latent.pow(2) - logvar_latent.exp(), dim=1)
    KLD = kl_divergence(z_sample, mu_latent, logvar_latent.exp())
    return torch.mean(-loss_rec + beta * KLD)


def loss_vae_lognorm(image_batch,decoded_data, mu_x, logvar_x, mu_latent, logvar_latent, beta=1.):
    '''Función que realiza el cómputo de la función de pérdidas bajo una asunción de gaussianidad. '''
    #todo


def loss_litvae(targets, predictions, mean, log_variance, beta=1.): 
    # x, x_recon, z_mean, z_logvar, beta=1.
    mse = F.mse_loss(predictions, targets, reduction='sum')
    log_sigma_opt = 0.5 * mse.log()
    r_loss = 0.5 * torch.pow((targets - predictions) / log_sigma_opt.exp(), 2) + log_sigma_opt
    r_loss = r_loss.sum()
    kl_loss = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
    return r_loss + beta*kl_loss