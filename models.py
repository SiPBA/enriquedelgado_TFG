import torch
from torch import nn

###################################################################################################
#                             PRIMERA PARTE: ARQUITECTURA DEL ENCODER                             #
###################################################################################################

class Encoder(nn.Module):
   
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        ### Seccion Convolucional
        self.encoder_cnn = nn.Sequential(
            #Conv3d(canales de input, canales de output, tamaño kernel, etc.)
            nn.Conv3d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True), #Deja pasar solo los valores >0
            nn.Conv3d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm3d(16), #Restringe los valores de las salidas 
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Capa de aplanado
        self.flatten = nn.Flatten(start_dim=1)
        
        ### Seccion Lineal
        self.encoder_lin = nn.Sequential(
            nn.Linear(11*15*11*32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

###################################################################################################
#                             SEGUNDA PARTE: ARQUITECTURA DEL DECODER                             #
###################################################################################################
        
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 11*15*11*32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=( 32, 11, 15, 11))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d( 32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.ConvTranspose3d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x