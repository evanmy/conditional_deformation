'''Input Libraries'''
import os
import torch
import sklearn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data as data_utils

from tqdm import tqdm
from functions import models as m
from argparse import ArgumentParser
from functions import dataset as data
from torch.utils.data import DataLoader
from functions import training_tools as tt
from functions import spatial_transformer as st

def eval_parser():

    parser = ArgumentParser()

    parser.add_argument("gpus", type=str,
                        help="Which gpus to use?")
    
    parser.add_argument("file", type=str,
                        help="File path + name")

    return  parser.parse_args()

def run(loader,
        template,
        models,
        optimizer,
        epoch,
        mode,
        args,
        PATH,
        subsample_loader):

    """Define Variables"""
    eps = 1e-12
    FeaturesNet = models[0]
    
    dataset = loader[0]
    sampler = loader[1]
    
    """Choose samples"""
    if mode == 'train':
        FeaturesNet.train()
        suffix = '_train'
    else:
        FeaturesNet.eval()
        suffix = '_eval'

    """Train"""    
    predicted = []
    attribute = np.load('./files/female_age_attribute.npy')
    for i,y in tqdm(enumerate(attribute)):
        """Load Data"""
        y = y.reshape(1, -1)
        _y = y.copy()
        name = 'sex{}_age{}_cdr{}'.format(_y[0,0], _y[0,1], _y[0,2])
        y[:,1] = (y[:,1]-args.age_mean)/args.age_std
        y = torch.from_numpy(y.astype('float32')).cuda()
        template = template.float().cuda().detach()
        
        """Predict deformation field"""
        optimizer.zero_grad()
        u = FeaturesNet(y)
        
        # Square and scale
        batch_size, chs, dim1, dim2, dim3 = template.shape
        int_steps = 7
        u = u / (2 ** int_steps)
        x = st.grid(batch_size,
                    dim1, 
                    dim2,
                    dim3).permute(0, 4, 1, 2, 3)
        phi = x + u

        for _ in range(int_steps):
            phi = F.grid_sample(phi,
                                grid=phi.permute(0, 2, 3, 4, 1),
                                mode='bilinear',
                                padding_mode='zeros')
            
        deformed_template = F.grid_sample(template,
                                          grid=phi.permute(0, 2, 3, 4, 1), 
                                          mode='bilinear',
                                          padding_mode='zeros')
        
        deformed_template = (vis.argmax_ch(deformed_template)).float()
        deformed_template = deformed_template.detach().cpu().numpy()
        np.save(PATH+str(i), deformed_template)
    return None

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from functions import visualization as vis
    
    eval_args = eval_parser()

    """Select GPU"""
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= eval_args.gpus
    summary = torch.load(eval_args.file)
    args = summary['args']
    
    """Load Data"""
    soi = data.get_subject(path='./files/phenotype.csv',
                           use_healthy= False)
    dataset = data.load_oasis_data(soi)
    atlas_path = './files/atlas.npz'
    template = data.get_affine_atlas(atlas_path)    
    
    """Making Model"""
    FeaturesNet = m.GenerativeFeaturesNet(in_features=3)
    FeaturesNet = torch.nn.DataParallel(FeaturesNet)
    FeaturesNet.cuda()

    optimizer = torch.optim.Adam(FeaturesNet.parameters(),
                                 lr= args.lr)

    
    FeaturesNet.load_state_dict(summary['FeaturesNet'])
    
    params = list(FeaturesNet.parameters())                            
    optimizer = torch.optim.Adam(params,
                                 lr= args.lr)
    PATH = './images/healthy_female_age_progression/'
    
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        raise Exception('Path exists :(')
        
    """Eval"""
    with torch.no_grad():
        mode = 'eval'
        stat = run(loader= [dataset, None],
                   template= template,
                   models= [FeaturesNet],
                   optimizer= optimizer,
                   epoch= 888,
                   mode= mode,
                   args= args,
                   PATH= PATH,
                   subsample_loader= None)
