# -*- coding: utf-8 -*-
from models.main_stage_1 import Conv_k3, Upblock,G_Stage1, CondAugment_Model
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, plane):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            Conv_k3(plane, plane),
            nn.BatchNorm2d(plane),
            nn.ReLU(True),
            Conv_k3(plane, plane),
            nn.BatchNorm2d(plane)
        )
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        tmp = x
        o = self.block(x)
        o = o + tmp
        return self.relu(o)
    
class G_Stage2(nn.Module):
    def __init__(self, G_Stage1):
        super(G_Stage2, self).__init__()
        self.G1 = G_Stage1
        self.CA = CondAugment_Model()
        for p in self.G1.parameters():
            p.requires_grad = False
        self.encoder = nn.Sequential(
            Conv_k3(3, 128),
            nn.ReLU(True),
            nn.Conv2d(128, 128 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(True),
            nn.Conv2d(128 * 2, 128 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 4),
            nn.ReLU(True))
        self.combine = nn.Sequential(
            Conv_k3(640, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.residual = nn.Sequential(
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512)
        )
        self.decoder = nn.Sequential(
            Upblock(512,256),
            Upblock(256,128),
            Upblock(128,64),
            Upblock(64,32),
            Conv_k3(32,3),
            nn.Tanh()
        )
        
    def forward(self, noise, emb):
        init_image, _, _ = self.G1(noise, emb)
        encoded = self.encoder(init_image)
        
        cond, m, s = self.CA(emb)
        cond = cond.view(-1, 128, 1, 1)
        cond = cond.repeat(1, 1, 16, 16)
        
        encoded_cond = torch.cat([encoded, cond],1)
        img_feature = self.combine(encoded_cond)
        img_feature = self.residual(img_feature)
        img = self.decoder(img_feature)
        
        return init_image, img, m, s
        
class D_Stage2(nn.Module):
    def __init__(self):
        super(D_Stage2, self).__init__()
        self.img_encoder1 = nn.Sequential(
            # start 3 x 256 x 256
            nn.Conv2d(3, 128, 4, 2, 1, bias=False), #=> 128 x 128 x 128
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), #=> 256 x 64 x 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), #=> 512 x 32 x 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), #=> 1024 x 16 x 16
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(1024, 2048, 4, 2, 1, bias=False), #=> 2048 x 8 x 8
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(2048, 4096, 4, 2, 1, bias=False), #=> 4096 x 4 x 4
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(0.2, True),
            
            Conv_k3(4096, 2048), # 2048 x 4 x 4
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, True),
            Conv_k3(2048, 1024), # 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True)
        )

        self.img_encoder2 =  nn.Sequential(
            Conv_k3(in_p=1024+128, out_p=1024),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=4),
            nn.Sigmoid()
        )
        
    def forward(self, img, cond):
        img_feature = self.img_encoder1(img)
        cond = cond.view(-1, 128 , 1, 1)
        cond = cond.repeat(1, 1, 4, 4)
        image_with_cond = torch.cat((img_feature, cond), 1)
        return self.img_encoder2(image_with_cond).view(-1)