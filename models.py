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

        # Capas fully connected para computar mu y sigma
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
            nn.ConvTranspose3d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )

        # PRUEBAS PARA OBTENER LA MEDIA Y LA VARIANZA DE LOS PIXELES
        # self.fc3 = nn.Linear(128, encoded_space_dim)
        # self.fc4 = nn.Linear(128, encoded_space_dim)
        
    def sample(self, mu, logvar):
        var = torch.exp(0.5 * logvar)
        # Reparametrización Trick para poder computar los gradientes y hacer backpropagation
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
        #------------------------------------------------------------
        # PRUEBAS PARA OBTENER LA MEDIA Y LA VARIANZA DE LOS PIXELES
        # # MÉTODO 1:
        #aux = x.view(x.size(0), -1)
        #mu_x = self.fc3(aux)
        #logvar_x = self.fc4(aux)
        # # MÉTODO 2:
        mu_x = x.mean(dim=(2,3,4), keepdim=True)
        logvar_x = x.var(dim=(2,3,4), keepdim=True).log()
        #------------------------------------------------------------
        return x, mu_x, logvar_x

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x, mu_x, logvar_x = self.decode(z)
        return x, mu, logvar, mu_x, logvar_x

#--------------------------------------------------------------------------------------------------
