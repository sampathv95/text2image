import os
import torch
from dataset import BirdDataset
from models.main_stage_1 import G_Stage1, D_Stage1
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.optim as optim
from helpers import KL_loss, weights_init, cal_D_loss, cal_G_loss

CUDA = True
cond_dim = 128
df_dim = 128
gf_dim = 128
z_dim = 100
emb_dim = 1024

def train_stage1():
    print('in')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    transform = transforms.Compose([
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = BirdDataset(dataDir = './data/bird_stack/', split='train', transform=transform)
    tr_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    netG = G_Stage1().to(device)
    netG.apply(weights_init)
    netD = D_Stage1().to(device)
    netD.apply(weights_init)
    lr = 0.0002
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.rand(batch_size, z_dim, 1, 1).to(device)

    real_labels = (torch.FloatTensor(batch_size).fill_(1)).to(device)
    fake_labels = (torch.FloatTensor(batch_size).fill_(0)).to(device)
    
    if not (os.path.isdir('./Result_stage1/')):
        os.makedirs('Result_stage1')
    num_epoch = 600
    iters = 0
    for epoch in range(num_epoch):
        if epoch % 100 == 0 and epoch > 0:
            lr = lr*0.5
            for param_group in optG.param_groups:
                param_group['lr'] = lr
            for param_group in optD.param_groups:
                param_group['lr'] = lr
        for i, data in enumerate(tr_loader,0):
            real_imgs, encoded_caps = data
            real_imgs = real_imgs.to(device)
            encoded_caps = encoded_caps.to(device)

            ##update discriminator
            netD.zero_grad()
            # generate fake image
            noise = torch.rand(batch_size, z_dim, 1, 1).to(device)
            fake_imgs, m, s = netG(noise, encoded_caps)
            errD, errD_real, errD_wrong, errD_fake = cal_D_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, m)
            errD.backward()
            optD.step()

            ##update generator
            netG.zero_grad()
            errG = cal_G_loss(netD, fake_imgs, real_labels, m)
            errG += errG + KL_loss(m,s)
            errG.backward()
            optG.step()

            if i%1 == 0:
                 print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_D_R: %.4f\tLoss_D_W: %.4f\tLoss_D_F %.4f'
                      % (epoch, num_epoch, i, len(tr_loader),
                         errD.item(), errG.item(), errD_real, errD_wrong, errD_fake))
        if epoch%50==0:
            with torch.no_grad():
                fake, _, _  = netG(fixed_noise, encoded_caps)
                fig = plt.figure(figsize=(10,10))
                grid = make_grid(fake.detach().cpu(), nrow=8, normalize=True).permute(1,2,0).numpy()
                plt.imshow(grid)
                fig.savefig('./Result_stage1/epch-{}.png'.format(epoch))

    torch.save(netG.state_dict(), './Result_stage1/netG_epoch_600.pth')
    torch.save(netD.state_dict(), './Result_stage1/netD_epoch_last.pth')

if __name__ == '__main__':
    train_stage1()