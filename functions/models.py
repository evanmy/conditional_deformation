import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class FeaturesNet(nn.Module):
    def __init__(self, in_features):
        super(FeaturesNet, self).__init__()
        self.dense = nn.Linear(in_features, 128*5*6*7, bias=True)
        self.tconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)
        self.tconv5 = nn.ConvTranspose3d(8, 3, kernel_size=2, stride=2)
        
#         self.tconv5.weight.data.fill_(0.0)
#         self.tconv5.bias.data.fill_(0.0)
        
    def forward(self, x):
        batch_size = x.shape[0]
        out = self.dense(x)
        out = F.relu(out)
        out = out.view(batch_size,128,5,6,7)
        
        out = self.tconv1(out)
        out = F.relu(out)
        
        out = self.tconv2(out)
        out = F.relu(out)        
        
        out = self.tconv3(out)
        out = F.relu(out)  
        
        out = self.tconv4(out)
        out = F.relu(out)  

        return self.tconv5(out) 

class GenerativeFeaturesNet(nn.Module):
    def __init__(self, in_features):
        super(GenerativeFeaturesNet, self).__init__()
        self.dense1 = nn.Linear(in_features, 128*5*6*7, bias=True)
        self.dense2 = nn.Linear(in_features, 128*5*6*7, bias=True)
        self.tconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)
        self.tconv5 = nn.ConvTranspose3d(8, 3, kernel_size=2, stride=2)
        
#         self.tconv5.weight.data.fill_(0.0)
#         self.tconv5.bias.data.fill_(0.0)
        
    def forward(self, x):
        batch_size = x.shape[0]
        mean = self.dense1(x)
        mean = F.relu(mean)
        mean = mean.view(batch_size,128,5,6,7)
        
        std = self.dense2(x)
        std = F.relu(std)
        std = std.view(batch_size,128,5,6,7)        
        
        if self.training:
            out = torch.randn_like(std)*std + mean 
        else:
            out = mean
            
        out = self.tconv1(out)
        out = F.relu(out)
        
        out = self.tconv2(out)
        out = F.relu(out)        
        
        out = self.tconv3(out)
        out = F.relu(out)  
        
        out = self.tconv4(out)
        out = F.relu(out)  

        return self.tconv5(out)     
    
def grid(batch_size, dim1, dim2, dim3, use_cuda=True):
    lin1 = torch.linspace(-1, 1, dim1)
    lin2 = torch.linspace(-1, 1, dim2)
    lin3 = torch.linspace(-1, 1, dim3)
    mesh1, mesh2, mesh3 = torch.meshgrid(lin1, lin2, lin3)
    mesh1 = mesh1.unsqueeze(3)
    mesh2 = mesh2.unsqueeze(3)
    mesh3 = mesh3.unsqueeze(3)
    grid = torch.cat((mesh3, mesh2, mesh1),3)
    grid = grid.unsqueeze(0)
    grid = grid.repeat(batch_size,1,1,1,1)
    grid = grid.float()
    if use_cuda:
        grid = grid.cuda()
    return grid

def one_layer_decoder(input_ch, out_ch):
    chs = [input_ch, out_ch]
    conv3d = nn.Conv3d(chs[0], chs[1], kernel_size=3, padding=1)
    layers = [conv3d, torch.nn.Sigmoid()]
    return nn.Sequential(*layers)  