from __future__ import print_function

#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import time
import copy
import tqdm

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# root directory for dataset
data_root = os.path.join(os.getcwd(), "data/vs_train")

# number of workers
workers = 4
# batch size during training
batch_size = 32
# spatial size of training images
image_size = 200
# number of channels
nc = 3
# size of z latent  vector
nz = 100
# Size of feature maps in encoder
ngf = 128
# Size of feature maps in deconder
ndf = 128
# number of epochs
num_epoch = 50
# learning rate
lr = 0.0002
# beta in Adam optimizers
beta1 = 0.5
# number of gpu
ngpu = 1


def get_input(data_root):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.CenterCrop(200),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.CenterCrop(200),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val_y": transforms.Compose(
            [
                transforms.CenterCrop(200),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "train_y": transforms.Compose(
            [
                transforms.CenterCrop(200),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    data_dir = data_root
    image_datasets = {
        x: dset.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val", "val_y", "train_y"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size,num_workers=workers
        )
        for x in ["train", "val", "val_y", "train_y"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "val_y", "train_y"]}
    device = "cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
    return dataloaders, dataset_sizes, device


print(os.getcwd())

dataloaders, dataset_sizes, device = get_input(data_root)
'''real_batch = next(iter(dataloaders["train"]))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(
            real_batch[0].to(device)[:200], padding=2, normalize=True
        ).cpu(),
        (1, 2, 0),
    )
)
plt.pause(1)'''

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Img2Img(nn.Module):
    def __init__(self, ngpu):
        super(Img2Img, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input a 200*200 image
            # encoder like network
            # 200*200*3 -> 100*100*128
            nn.Conv2d(nc, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 100*100*128 -> 50*50*128*2
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 50*50*128*2 -> 25*25*128*4
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 25*25*128*4 -> 12*12*128*8
            nn.Conv2d(ngf * 4, ngf * 8, 5, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 12*12*128*8 -> 6*6*128*16
            nn.Conv2d(ngf * 8, ngf * 16, 4, 2, 1),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # 6*6*128*16-> 1*1*100
            nn.Conv2d(ngf * 16, nz, 6, 1, 0),
            nn.BatchNorm2d(nz),
            nn.ReLU(True),
            # decoder
            # 1*1*100 -> 6*6*128*16
            nn.ConvTranspose2d(nz, ngf * 16, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # 6*6*128*16 -> 12*12*128*8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 12*12*128*8 -> 25*25*128*4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 25*25*128*4 -> 50*50*128*2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 50*50*128*2 -> 100*100*128
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 100*100*128 -> 200*200*3
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 200*200*3 -> 100*100*128
            nn.Conv2d(nc, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 100*100*128 -> 50*50*128*2
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 50*50*128*2 -> 25*25*128*4
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 25*25*128*4 -> 12*12*128*8
            nn.Conv2d(ngf * 4, ngf * 8, 5, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 12*12*128*8 -> 6*6*128*16
            nn.Conv2d(ngf * 8, ngf * 16, 4, 2, 1),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 6*6*128*16-> 1*1*100
            nn.Conv2d(ngf * 16, 1, 6, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)





# Create the generator
netG = Img2Img(ngpu).to(device)

# Handle multi-gpu if desired
if (device == "cuda") and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print('='*20)
print('Model of Generator')
print(netG)
# Initialize MSELoss function
criterion = nn.L1Loss()
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device == "cuda") and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)
# Print the model
print('='*20)
print('Model of Discriminator')
print(netD)

# Initialize BCELoss function
criterionD = nn.BCELoss()
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
scheduler = lr_scheduler.StepLR(optimizerG, step_size=7, gamma=0.1)
# Training Loop


def train_Gan(netG,netD,criterion, criterionD, optimizerG, optimizerD, dataloaders, dataset_sizes, num_epoch):
    G_losses = []
    D_losses = []
    iters = 0
    print("Starting Training Loop...")
    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch + 1, num_epoch))
        print("--" * 10)

        running_loss = 0.0
        print(dataloaders['train'])
        iter_inputs = iter(dataloaders['train'])
        iter_labels = iter(dataloaders['train_y'])
        total_batchs = len(iter_inputs)
        i = 0
        while True:
            try:
                inputs = next(iter_inputs)[0].to(device)
                labels = next(iter_labels)[0].to(device)
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # format batch
                real_cpu = labels
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                noise = torch.rand(b_size, dtype=torch.float, device=device)
                label = label-noise
                # Forward pass real batch through D
                outputD = netD(real_cpu).view(-1)
               
                # Calculate loss on all-real batch
                errD_real = criterion(outputD, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = outputD.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                fake = netG(inputs)
                label.fill_(fake_label)
                noise = torch.rand(b_size, dtype=torch.float, device=device)
                label = label+noise
                # Classify all fake batch with D
                outputD = netD(fake.detach()).view(-1)
                
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(outputD, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = outputD.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                noise = 0.5*torch.rand(b_size, dtype=torch.float, device=device)
                label = label-noise
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f[ %.4f\ %.4f]\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, num_epoch, i, len(dataloaders),
                            errD.item(), errD_real, errD_fake, errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                i+=1 
                iters += 1

            except StopIteration:
                    break
    return netG, netD



def train_model(model,criterion, optimizerG, scheduler, dataloaders, dataset_sizes, num_epoch):
    since = time.time()

    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch + 1, num_epoch))
        print("-" * 10)
        for phase in  ['train' ,'val']:
            running_loss = 0.0
            iter_inputs = iter(dataloaders[phase])
            iter_labels = iter(dataloaders[phase+'_y'])
            if phase == 'train':
                par = tqdm.tqdm(total=len(iter_inputs), ncols=100)
                model.train()
            else:
                model.eval()
            while True:
                try:
                    inputs = next(iter_inputs)[0].to(device)
                    labels = next(iter_labels)[0].to(device)
                
                    # zero the parameter gradients
                    optimizerG.zero_grad()

                    #forward
                    with torch.set_grad_enabled(phase=='train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            par.update(1)
                            loss.backward()
                            optimizerG.step()
                    
                    running_loss += loss.item()*inputs.size(0)
                except StopIteration:
                    break
            par.close()
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss/dataset_sizes[phase]
            print()
            print('{} loss: {:.4f}'.format(phase, epoch_loss))
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed %60))

    return model

def svimg(model, dataloaders, path):
    with torch.no_grad():
        i = 0
        for inputs in iter(dataloaders['train']):
                inputs = inputs[0].to(device)
                outputs = model(inputs)
                for o in iter(outputs):
                    save_image(o[0], path+ str(i)+'.jpg')
                    i+=1


#model_ft, _ =train_Gan(netG,netD,criterion, criterionD, optimizerG, optimizerD, dataloaders, dataset_sizes, num_epoch)
model_ft = train_model(netG,criterion, optimizerG, scheduler, dataloaders, dataset_sizes, num_epoch)
svimg(model_ft, dataloaders, data_root+'/output/')