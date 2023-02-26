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
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=( 32, 11, 15, 11)),
            nn.ConvTranspose3d( 32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            # En la capa de abajo se almacena en un canal la media y en el otro el logvar de cada pixel
            nn.ConvTranspose3d(8, 2, 3, stride=2, padding=1, output_padding=1),
        )

        # Capas convolucionales para computar la media y el logaritmo de la varianza de los pixeles de la imagen recontruida
        # self.cv1 = nn.Sequential(
        #     nn.Conv3d(1, 8, 3, stride=2, padding=1),
        #     nn.ConvTranspose3d(8, 1, 3, stride=2, padding=1, output_padding=1)
        # )
        # self.cv2 = nn.Sequential(
        #     nn.Conv3d(1, 8, 3, stride=2, padding=1),
        #     nn.ConvTranspose3d(8, 1, 3, stride=2, padding=1, output_padding=1)
        # )

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
        #mu_x = self.cv1(x)
        #logvar_x = self.cv2(x)
        mu_x = x[:,0,:,:,:]
        logvar_x = x[:,1,:,:,:]
        return mu_x, logvar_x

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x, mu_x, logvar_x = self.decode(z)
        return x, mu, logvar, mu_x, logvar_x

#--------------------------------------------------------------------------------------------------
