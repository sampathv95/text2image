import torch
import torch.nn as nn
import torch.optim as optim

def KL_loss(mean, sigma):
    temp = 1+sigma+((-1)*((mean*mean)+sigma))
    return torch.mean(temp)*(-0.5)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def cal_G_loss(netD, fake_imgs, real_labels, cond):
    criterion = nn.BCELoss()
    cond = cond.detach()
    fake_prob = netD(fake_imgs, cond)
    errD_fake = criterion(fake_prob, real_labels)
    return errD_fake

def cal_D_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, cond):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = cond.detach()
    fake = fake_imgs.detach()

    real_prob = netD(real_imgs, cond)
    fake_prob = netD(fake, cond)
    wrong_prob = netD(real_imgs[:(batch_size-1)], cond[1:])

    errD_real  = criterion(real_prob, real_labels)
    errD_wrong = criterion(wrong_prob, fake_labels[1:])
    errD_fake= criterion(fake_prob, fake_labels)
    
    errD = errD_real + (errD_fake+errD_wrong)*0.5

    return errD, errD_real.item(), errD_wrong.item(), errD_fake.item()