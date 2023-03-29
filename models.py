###################################################################################################
# DEFINICIÓN DE LA ARQUITECTURA DE LOS MODELOS:                                                   #
#--------------------------------------------------------------------------------------------------
# Este script define dos arquitecturas para dos modelos diferentes:                               #
#--------------------------------------------------------------------------------------------------
# 1. Arquitectura de un AutoEncoder Convolucional (CAE). Compuesto por una sección convolucional  #
# con 3 capas convolucionales, una capa de aplanado y una sección lineal compuesta de dos capas   #
# lineales.                                                                                       #
#--------------------------------------------------------------------------------------------------
# 2. Arquitectura de un AutoEncoder Convolucional Variacional (CVAE). Compuesto igual que el CAE. #
#--------------------------------------------------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
###################################################################################################
#                             PRIMER MODELO: AUTOENCODER CONVOLUCIONAL                            #
###################################################################################################
class CAE_3D(nn.Module):
   
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
    
        self.encoder_CAE = nn.Sequential(
            ### Seccion Convolucional
            nn.Conv3d(1, 8, 3, stride=2, padding=1), #Conv3d(canales de input, canales de output, tamaño kernel, etc.)
            nn.ReLU(True), #Deja pasar solo los valores >0
            nn.Conv3d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm3d(16), #Restringe los valores de las salidas 
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            ### Capa de aplanado
            nn.Flatten(start_dim=1),
            ### Seccion Lineal
            nn.Linear(11*15*11*32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

        self.decoder_CAE = nn.Sequential(
            ### Seccion Lineal
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 11*15*11*32),
            nn.ReLU(True),
            ### Capa de desaplanado
            nn.Unflatten(dim=1, unflattened_size=( 32, 11, 15, 11)),
            ### Capa de convolución transpuesta
            nn.ConvTranspose3d( 32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.ConvTranspose3d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )           
            
    def encode(self, x):
        z = self.encoder_CAE(x)
        return z
    
    def decode(self, z):
        x = torch.sigmoid(self.decoder_CAE(z))
        return x

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x
#--------------------------------------------------------------------------------------------------

###################################################################################################
#                             SEGUNDO MODELO: AUTOENCODER VARIACIONAL                             #
###################################################################################################

class CVAE_3D(nn.Module):
   
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        self.encoder_CVAE = nn.Sequential(
            ### Sección convolucional
            nn.Conv3d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True), 
            nn.Conv3d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm3d(16), 
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            ### Capa de aplanado
            nn.Flatten(start_dim=1),
            ### Sección lineal
            nn.Linear(11*15*11*32, 128),
            nn.ReLU(True)
        )

        # Capas fully connected para computar la media y el logaritmo de la varianza del espacio latente
        self.fc1 = nn.Linear(128, encoded_space_dim)
        self.fc2 = nn.Linear( 128, encoded_space_dim)

        self.decoder_CVAE = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 11*15*11*32),
            nn.Unflatten(dim=1, unflattened_size=( 32, 11, 15, 11)),
            nn.ReLU(True),
            nn.ConvTranspose3d( 32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),            
            nn.ConvTranspose3d(8, 2, 3, stride=2, padding=1, output_padding=1),
            # En la capa de abajo se almacena en un canal la media y en el otro el logvar de cada pixel
            nn.Conv3d(2,2,3, stride=1, padding=1)
        )


    def sample(self, mu, logvar):
        var = torch.exp(0.5 * logvar)
        # Truco de reparametrización para poder computar los gradientes y hacer la propagación hacia atrás
        eps = torch.randn_like(var)
        z = mu + var * eps
        return z

    def bottleneck(self, h):
        mu = self.fc1(h)
        logvar = self.fc2(h)
        z = self.sample(mu, logvar)
        return z, mu, logvar
    
    def encode(self, x):
        h = self.encoder_CVAE(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        x = self.decoder_CVAE(z)
        mu_x = x[:,0,:,:,:]
        logvar_x = x[:,1,:,:,:]
        return mu_x, logvar_x

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        mu_x, logvar_x = self.decode(z)
        return z, mu, logvar, mu_x, logvar_x

#--------------------------------------------------------------------------------------------------
# OTRO EJEMPLO DE ENCODER-DECODER, BASADO EN ALEXNET.

class Conv3DVAEEncoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(Conv3DVAEEncoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 8 * 6, 512)
        self.fc2_mean = nn.Linear(512, latent_dim)
        self.fc2_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        # Encode
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256 * 6 * 8 * 6)
        x = F.relu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_logvar = self.fc2_logvar(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def kl_loss(self, mu, logvar, reduction='sum'):
        kl_batch =  -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        if reduction=='sum':
            return kl_batch.sum()
        else:
            return kl_batch.mean()
    
class Conv3DVAEDecoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(Conv3DVAEDecoder, self).__init__()

        # Decoder
        self.fc1 = nn.Linear(latent_dim, 256 * 6 * 8 * 6)
        self.conv1 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Decode
        x = F.relu(self.fc1(x))
        x = x.view(-1, 256, 6, 8, 6)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.sigmoid(self.conv4(x))
        return x
    
    def recon_loss(self, targets, predictions, reduction='sum'):
        # From arXiv calibrated decoder: arXiv:2006.13202v3
        # D is the dimensionality of x. 
        r_loss = F.mse_loss(predictions, targets, reduction=reduction)
        # torch.pow(predictions-targets, 2).mean(dim=(1,2,3,4)) #+ D * self.logsigma
        return r_loss
    

class Conv3DVAEDecoderWParams(Conv3DVAEDecoder):

    def forward(self, x, params):
        # Decode
        assert x.shape[1] == params.shape[1]
        x = torch.cat((x, params), 1)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 256, 6, 8, 6)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.sigmoid(self.conv4(x))
        return x
    
class Conv3DVAEEncoderDilation(nn.Module):
    def __init__(self, latent_dim=3):
        super(Conv3DVAEEncoderDilation, self).__init__()
        self.latent_dim = latent_dim

        # use dilated convolutions for the first two layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, dilation=2, stride=2, padding=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, dilation=2, stride=2, padding=2)

        # use standard convolutions with stride for the next two layers
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)

        # compute the size of the output from the convolutional layers
        self.conv_output_size = 256 * 6 * 8 * 6

        # use fully-connected layers to obtain the mean and log-variance of the latent distribution
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, self.latent_dim)
        self.fc3 = nn.Linear(512, self.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = x.view(-1, self.conv_output_size)
        x = nn.functional.relu(self.fc1(x))
        mu = self.fc2(x)
        logvar = self.fc3(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def kl_loss(self, mu, logvar, reduction='sum'):
        kl_batch =  -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        if reduction=='sum':
            return kl_batch.sum()
        else:
            return kl_batch.mean()


class Conv3DVAEDecoderAlt(nn.Module):
    def __init__(self, latent_dim=3, init_sigma=0.1):
        super(Conv3DVAEDecoderAlt, self).__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(latent_dim, 256 * 6 * 8 * 6)
        self.conv1 = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
        self.logsigma = nn.Parameter(torch.Tensor([init_sigma]))
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 6, 8, 6)
        x = nn.Upsample(scale_factor=(2,2,2), mode='nearest')(x)
        x = F.relu(self.conv1(x))
        x = nn.Upsample(scale_factor=(2,2,2), mode='nearest')(x)
        x = F.relu(self.conv2(x))
        x = nn.Upsample(scale_factor=(2,2,2), mode='nearest')(x)
        x = F.relu(self.conv3(x))
        x = nn.Upsample(scale_factor=(2,2,2), mode='nearest')(x)
        x = self.conv4(x)
        return x
    
    def recon_loss_calibrated(self, targets, predictions, D=1):
        # From arXiv calibrated decoder: arXiv:2006.13202v3
        # D is the dimensionality of x. 
        r_loss = torch.pow(predictions-targets, 2).mean(dim=(1,2,3,4))*D/(2*self.logsigma.exp()) #+ D * self.logsigma
        return r_loss.sum()


class Gen3DVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim=20, logsigma=1.0):
        super().__init__()
        self.latent_dim = latent_dim

        self.encode = encoder(latent_dim)
        self.decode = decoder(latent_dim, logsigma)

    def forward(self, x):
        # Encode
        z, z_mean, z_logvar = self.encode(x)

        # Decode
        x_recon = self.decode(z)

        return z, z_mean, z_logvar, x_recon
    
    def loss_function(self, x, x_recon, z_mean, z_logvar, beta=1., reduction='sum'):
        kl_loss = self.encode.kl_loss(z_mean, z_logvar, reduction=reduction)
        recon_loss = self.decode.recon_loss_calibrated(x, x_recon, reduction=reduction)
        return recon_loss + beta*kl_loss, recon_loss, kl_loss


class Conv3DVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim=20):
        super(Conv3DVAE, self).__init__()
        self.latent_dim = latent_dim

        self.encode = encoder(latent_dim)
        self.decode = decoder(latent_dim)

    def forward(self, x):
        # Encode
        z, z_mean, z_logvar = self.encode(x)

        # Decode
        x_recon = self.decode(z)

        return z, z_mean, z_logvar, x_recon

    def loss_function(self, x, x_recon, z_mean, z_logvar, beta=1., reduction='sum'):
        kl_loss = self.encode.kl_loss(z_mean, z_logvar, reduction=reduction)
        recon_loss = self.decode.recon_loss(x, x_recon, reduction=reduction)
        return recon_loss + beta*kl_loss, recon_loss, kl_loss


class Conv3DVAEWParams(Conv3DVAE):

    def forward(self, x, params):
        # Encode
        z, z_mean, z_logvar = self.encode(x)

        # Decode
        x_recon = self.decode(z, params)

        return z, z_mean, z_logvar, x_recon


    # def loss_function(self, x, x_recon, z_mean, z_logvar, beta=1.):
    #     # Reconstruction loss
    #     # reconstruction_loss = F.mse_loss(x_recon, x)
    #     mse_loss = torch.sum((x_recon-x)**2, dim=(1,2,3,4))

    #     # # KL divergence loss
    #     # kl_divergence_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    #     # covariance_loss = torch.sum(torch.exp(z_logvar), dim=(1))
    #     # log_likelihood = -0.5 * (mse_loss + covariance_loss)
    #     kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)

    #     # # Total loss
    #     # total_loss = reconstruction_loss + kl_divergence_loss
    #     total_loss = torch.mean(mse_loss + beta * kl_divergence)

    #     return total_loss