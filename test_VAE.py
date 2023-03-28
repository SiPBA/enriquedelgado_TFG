#%% 
import torch
from torch.utils.data import DataLoader
from loader import HDFImageDataset
import models
from utils import *
from tensorboardX import SummaryWriter
import torchvision

# from train import *
# from image_norms import integral_norm

ruta = '/home/pakitochus/Universidad/Investigación/Databases/parkinson/PPMI_ENTERA/IMAGENES_TFG/'
# ruta = 'C:\TFG\IMAGENES_TFG/'
num_epochs = 200
modelo_elegido='genvae'
d = 20
lr = 1e-3 # 3e-4
batch_size = 16
PARAM_BETA = 100.
PARAM_LOGSIGMA = np.log(.1)
PARAM_NORM = 3
filename = f'Conv3DVAE_d{d}_BETA{int(PARAM_BETA)}_lr{lr:.0E}_bs{batch_size}_n{num_epochs}_norm{PARAM_NORM}'

train_dataset = HDFImageDataset('medical_images.hdf5', norm=PARAM_NORM)
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True, shuffle=False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# image_batch = train_dataset[0][0].unsqueeze(0)
# patno = train_dataset[0][1]
# year = train_dataset[0][2]

# vae = models.Gen3DVAE(encoder=models.Conv3DVAEEncoderDilation,
#                       decoder=models.Conv3DVAEDecoderAlt,
#                       latent_dim=d,
#                       logsigma=PARAM_LOGSIGMA)
vae = models.Conv3DVAE(encoder=models.Conv3DVAEEncoder,
                       decoder=models.Conv3DVAEDecoder,
                       latent_dim=d)

vae = vae.to(device)
vae = torch.compile(vae)

# encoder, decoder = models.Conv3DVAEEncoderDilation(latent_size=d), models.Conv3DVAEDecoderAlt(latent_dim=d)

#%% 
# vae.load_state_dict(torch.load(filename+'.pth'))
writer = SummaryWriter('runs/'+filename)
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
# optimizer = torch.optim.SGD(vae.parameters(), lr=1e-7)
# num_epochs = 300
start=0

train_loss = []
for e in range(start, num_epochs):
    vae.train()
    train_epoch = []
    kl_epoch = []
    recon_epoch = []
    for ix, image_batch in enumerate(tqdm(train_loader)): 
        optimizer.zero_grad()
        # Mueve el tensor a la GPU si está disponible
        patno, year = image_batch[1], image_batch[2]
        image_batch = image_batch[0].to(device)
        # forward pass:
        z, z_mean, z_logvar, x_recon = vae(image_batch)
        loss, recon_loss, kl_loss = vae.loss_function(image_batch, x_recon, z_mean, z_logvar, beta=PARAM_BETA)
        loss.backward()
        optimizer.step()
        if ix%5==0:
            guarda_imag(x_recon.detach(), patno, year, e, modelo_elegido, num_epochs, d)
        train_epoch.append(loss.item())
        kl_epoch.append(kl_loss.item())
        recon_epoch.append(recon_loss.item())
    train_loss.append(sum(train_epoch)/len(train_epoch))
    print(f'E: {e}, loss: {train_loss[-1]} [kl: {sum(kl_epoch)/len(kl_epoch)}], [Recon: {sum(recon_epoch)/len(recon_epoch)}]')
    writer.add_scalar('Loss/total', train_loss[-1], e)
    writer.add_scalar('Loss/KLD', sum(kl_epoch)/len(kl_epoch), e)
    writer.add_scalar('Loss/Recon', sum(recon_epoch)/len(recon_epoch), e)
    espacio_latente = df_latente(test_dataloader, vae, modelo_elegido, device)
    anima_latente(espacio_latente, e, modelo_elegido, num_epochs, d)
    label_img = x_recon[..., 10:-22, 40]
    grid = torchvision.utils.make_grid(x_recon[..., 40], nrow=int(np.sqrt(batch_size)))
    writer.add_embedding(z.detach(), label_img=label_img, global_step=e)
    writer.add_image('images', grid, global_step=e)
    # writer.add_graph(vae, x_recon)

writer.close()
# #%%
# z, mu, logvar = encoder(image_batch)
# #%% 
# x_recon = decoder(z)
# # %%
# kl_loss, recon_loss = encoder.kl_loss(mu, logvar), decoder.recon_loss_calibrated(image_batch, x_recon)
# # %%
#%% 
import matplotlib.pyplot as plt 
imdata = next(iter(train_loader))
patno, year = imdata[1], imdata[2]
updrs_tot = imdata[5]
image_batch = imdata[0].to(device)
vae.eval()
with torch.no_grad():
    z, z_mean, z_logvar, x_recon = vae(image_batch)
grid_orig = torchvision.utils.make_grid(image_batch[...,40], nrow=int(np.sqrt(batch_size)))
grid = torchvision.utils.make_grid(x_recon[...,40], nrow=int(np.sqrt(batch_size)))
fig, ax = plt.subplots(ncols=3,figsize=(15,4))
ax[0].imshow(grid_orig[0].cpu().numpy(), vmin=0, vmax=3)
ax[0].set_title('Images original')
ax[1].imshow(grid[0].detach().cpu().numpy(), vmin=0, vmax=2)
ax[1].set_title('Images recon')
dim = 0
ax[2].errorbar(z_mean[:,dim].detach().cpu(), updrs_tot, yerr=z_logvar[:,dim].detach().cpu().exp(), fmt='o')
ax[2].set_title('batch latent manifold')
ax[2].set_xlabel(f'Variable {dim}')
ax[2].set_ylabel('UPDRS totscore on')
# %% GUARDAR RESULTADOS
torch.save(vae.state_dict(), filename+'.pth')
espacio_latente.to_csv(filename+'.csv')
# %%
