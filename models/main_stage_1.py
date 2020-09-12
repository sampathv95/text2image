# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
        
CUDA = False
cond_dim = 128
df_dim = 128
gf_dim = 128
z_dim = 100
emb_dim = 1024

def Conv_k3(in_p, out_p, stride=1):
    return nn.Conv2d(in_p, out_p, kernel_size=3, stride=stride, padding=1, bias=False)

class Upblock(nn.Module):
    def __init__(self, inp, outp):
        super(Upblock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = Conv_k3(inp, outp)
        self.batch = nn.BatchNorm2d(outp)
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        x = self.up(x)
        x = self.relu(self.conv(x))
        x = self.batch(x)
        return x

class CondAugment_Model(nn.Module):
    def __init__(self):
        super(CondAugment_Model,self).__init__()
        self.fc = nn.Linear(emb_dim, cond_dim*2)
        self.relu = nn.ReLU(True)
        
    def convert(self, embed):
        x = self.relu(self.fc(embed))
        mean, sigma = x[:, :cond_dim], x[:, cond_dim:]
        return mean, sigma
    
    def forward(self, x):
        mean, sigma = self.convert(x)
        diag = torch.exp(sigma*0.5)
        if CUDA:
            normal_dis = (torch.FloatTensor(diag.size()).normal_()).cuda()
        else:
            normal_dis = (torch.FloatTensor(diag.size()).normal_())
        condition = (diag*normal_dis)+mean
        return condition, mean, sigma
        
class G_Stage1(nn.Module):
    def __init__(self):
        super(G_Stage1, self).__init__()
        self.CA = CondAugment_Model()
        self.fc = nn.Sequential(
            nn.Linear(in_features=228, out_features=128*8*4*4, bias=False),
            nn.BatchNorm1d(128*8*4*4),
            nn.ReLU(inplace=True)
        )
        self.img = nn.Sequential(
            Upblock(128*8,64*8),
            Upblock(64*8,32*8),
            Upblock(32*8,16*8),
            Upblock(16*8,8*8),
            Conv_k3(8*8, 3),
            nn.Tanh()
        )
        
    def forward(self, noise, emb):
        cond, mean, sigma = self.CA(emb)
        cond = cond.view(noise.size(0), cond_dim, 1, 1)
        x = torch.cat((noise, cond),1)
        x = x.view(-1, 228)
        o = self.fc(x)
        h_code = o.view(-1, 128*8, 4, 4)
        fake_img = self.img(h_code)
        return fake_img, mean, sigma

class D_Stage1(nn.Module):
    def __init__(self):
        super(D_Stage1, self).__init__()
        self.encoder1 = nn.Sequential(
            #c alucalation output size = [(input_size −Kernal +2Padding )/Stride ]+1
            # input is image 3 x 64 x 64  
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),# => 128 x 32 x 32 
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),# => 256 x 16 x 16
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),# => 512 x 8 x 8
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)# => 1024 x 4 x 4
        )
        self.encoder2 = nn.Sequential(
            Conv_k3(in_p=1024+128, out_p=1024),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=4),
            nn.Sigmoid()
        )
        
    def forward(self, img, cond):
        img_feature = self.encoder1(img)
        cond = cond.view(-1, 128 , 1, 1)
        cond = cond.repeat(1, 1, 4, 4)
        image_with_cond = torch.cat((img_feature, cond), 1)
        return self.encoder2(image_with_cond).view(-1)