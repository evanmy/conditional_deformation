import math
import torch
import numbers
import sklearn
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def dice_loss(pred, target):
    eps = 1
    assert pred.size() == target.size(), 'Input and target are different dim'
    
    if len(target.size())==4:
        n,c,x,y = target.size()
    if len(target.size())==5:
        n,c,x,y,z = target.size()

    target = target.view(n,c,-1)
    pred = pred.view(n,c,-1)
    
    num = torch.sum(2*(target*pred),2) + eps
    den = (pred*pred).sum(2) + (target*target).sum(2) + eps
    dice_loss = 1-num/den
    total_avg = torch.mean(dice_loss)
    return total_avg

class Sampler():
    '''Sample idx without replacement'''
    def __init__(self, idx):
        self.idx = idx 
        self.iterator = iter(sklearn.utils.shuffle(idx))

    def sequential(self):
        try:
            return next(self.iterator)
        except:
            self.iterator = iter(sklearn.utils.shuffle(self.idx))
            return next(self.iterator)
        
    def shuffle(self):
        self.iterator = iter(sklearn.utils.shuffle(self.idx))
        
def argmax_ch(input):
    '''
    Pick the most likely class for each pixel
    individual mask: each subjects 
    have different uniformly sample mask
    '''
    input = input.detach().cpu()
    batch_n, chs, xdim, ydim, zdim = input.size()

    # Enumarate the chs #
    # enumerate_ch has dimensions [batch_n, chs, xdim, ydim, zdim]

    arange = torch.arange(0,chs).view(1,-1, 1, 1, 1)
    arange = arange.repeat(batch_n, 1, 1, 1, 1).float()
    
    enumerate_ch = torch.ones(batch_n, chs, xdim, ydim, zdim)
 
    enumerate_ch = arange*enumerate_ch 

    classes = torch.argmax(input,1).float()
    sample = []
    for c in range(chs):
        _sample = enumerate_ch[:,c,:,:,:] == classes
        sample += [_sample.unsqueeze(1)]
    sample = torch.cat(sample, 1)
    
    return sample

def normalize_dim1(x):
    '''
    Ensure that dim1 sums up to zero for proper probabilistic interpretation
    '''
    normalizer = torch.sum(x, dim=1, keepdim=True)
    return x/normalizer

def cross_entropy(pred, y):
    cce = -1*torch.sum(y*pred,1)            #cross entropy
    cce = torch.sum(cce,(1,2,3))            #cce over all the dims
    cce = cce.mean()              
    return cce