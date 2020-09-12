import os
import torch
import torch.nn as nn
from dataset import BirdDataset
from main_stage_1 import G_Stage1
from main_stage_2 import G_Stage2, D_Stage2
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.optim as optim
from helpers import KL_loss, weights_init, cal_D_loss, cal_G_loss

def train_stage2():
    device = torch.device('cpu')
    # load dataset with size 256x256
    batch_size = 16
    transform = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = BirdDataset(dataDir = './data/bird_stack/', split='train', transform=transform, imgSize=256)
    tr_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    #load model Stage-I generator and put it into Stage-II generator
    G1 = G_Stage1()
    G1.load_state_dict(torch.load('./Result_stage1/netG_epoch_600.pth', map_location=torch.device('cpu')))
    G1.eval()
    netG = G_Stage2(G1)
    netG.apply(weights_init)
    netG = nn.DataParallel(netG).to(device)
    netD = D_Stage2()
    netD.apply(weights_init)
    netD = nn.DataParallel(netD).to(device)

    lr = 0.0002
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    # remove the parameter from Stage-I generator
    netG_param = []
    for p in netG.parameters():
        if p.requires_grad:
            netG_param.append(p)
    optG = optim.Adam(netG_param, lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.rand(batch_size, 100, 1, 1).to(device)
    
    if not (os.path.isdir('./Result_stage2/')):
        os.makedirs('Result_stage2')
    num_epoch = 600
    iters = 0
    for epoch in range(num_epoch+1):
        if epoch % 100 == 0 and epoch > 0:
            lr = lr*0.5
            for param_group in optG.param_groups:
                param_group['lr'] = lr
            for param_group in optD.param_groups:
                param_group['lr'] = lr
        train_epoch(epoch, batch_size, tr_loader, netG, netD, fixed_noise, optD, optG, device)

def train_epoch(epoch, batch_size, tr_loader, netG, netD, fixed_noise, optD, optG, device):

    real_labels = (torch.FloatTensor(batch_size).fill_(1)).to(device)
    fake_labels = (torch.FloatTensor(batch_size).fill_(0)).to(device)

    for i, data in enumerate(tr_loader,0):
        real_imgs, encoded_caps = data
        real_imgs = real_imgs.to(device)
        encoded_caps = encoded_caps.to(device)

        ##update discriminator
        netD.zero_grad()
        # generate fake image
        noise = torch.rand(batch_size, 100, 1, 1).to(device)
        init_img ,fake_imgs, m, s = netG(noise, encoded_caps)
        errD, errD_real, errD_wrong, errD_fake = cal_D_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, m)
        errD.backward()
        optD.step()

        ##update generator
        netG.zero_grad()
        errG = cal_G_loss(netD, fake_imgs, real_labels, m)
        errG += errG + KL_loss(m,s)
        errG.backward()
        optG.step()

        if i%50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_D_R: %.4f\tLoss_D_W: %.4f\tLoss_D_F %.4f'
                    % (epoch, 600, i, len(tr_loader),
                        errD.item(), errG.item(), errD_real, errD_wrong, errD_fake))
        
        return 
        if epoch%10==0:
            with torch.no_grad():
                _, fake, _, _  = netG(fixed_noise, encoded_caps)
                fig = plt.figure(figsize=(10,10))
                grid = make_grid(fake.detach().cpu(), nrow=8, normalize=True).permute(1,2,0).numpy()
                plt.imshow(grid)
                fig.savefig('./Result_stage2/epch-{}.png'.format(epoch))
        if epoch%25==0:
            torch.save(netG.state_dict(), './Result_stage2/netG2_epoch_{}.pth'.format(epoch))
            torch.save(netD.state_dict(), './Result_stage2/netD2_epoch_{}.pth'.format(epoch))

if __name__ == '__main__':
    train_stage2()