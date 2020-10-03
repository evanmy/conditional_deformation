'''Input Libraries'''
import os
import torch
import sklearn
import torch.nn.functional as F

import numpy as np
import torch.utils.data as data_utils

from tqdm import tqdm
from functions import models as m
from functions import dataset as data
from torch.utils.data import DataLoader
from functions.parser import train_parser
from functions import training_tools as tt
from functions import spatial_transformer as st

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
        sampler.shuffle()

    """Train"""
    running_loss = []
    for j, si in enumerate(range(subsample_loader)):
        """Load Data"""
        # Sample
        labeled_idx = sampler.sequential()
          
        # Preprocess
        mri, _, input, y = dataset[labeled_idx]
        y = y.values[:,[1,2,3]] #gender, Age, CDR
        y[:,1] = (y[:,1]-args.age_mean)/args.age_std
        
        input = input.cuda()
        y = torch.from_numpy(y.astype('float32')).cuda()
        template = template.float().cuda().detach()
        
        """Predict deformation field"""
        optimizer.zero_grad()
        u = FeaturesNet(y)
        
        # Square and scale
        batch_size, chs, dim1, dim2, dim3 = input.shape
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
        if args.use_dice:
            loss = tt.dice_loss(input, deformed_template)
        else:
            eps = 1e-8
            deformed_template = tt.normalize_dim1(deformed_template+eps)
            loss = tt.cross_entropy(torch.log(deformed_template), 
                                    input)
        if mode=='train':
            loss.backward()
            optimizer.step()
            
        running_loss += [loss.item()]
       
    loss = np.mean(running_loss)
    stat = [loss]
    
    if PATH is not None:
        vis.view_results(target_brain= mri,
                         recon_brain= mri,
                         pred_regs= deformed_template,
                         target_regs= input,
                         epoch= epoch,
                         suffix= suffix,
                         show_brain= False,
                         image_idx=0,
                         PATH=PATH)    
    return stat

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from functions import visualization as vis
    
    args = train_parser()

    """Select GPU"""
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus

    """Load Data"""
    soi = data.get_subject(path='./files/phenotype.csv',
                           use_healthy= False)
    dataset = data.load_oasis_data(soi)
    
    atlas_path = './files/atlas.npz'
    template = data.get_affine_atlas(atlas_path)    
       
    """Getting idx"""
    indices = len(dataset)
    indices = np.arange(len(dataset))
    indices = sklearn.utils.shuffle(indices, 
                                    random_state= args.seed)

    #args.selected_sbjs = train_set.vols_files # order of the files
    args.train_idx = indices[:2]
    args.test_idx = indices[2:]
    train_sampler = tt.Sampler(args.train_idx)
    
    args.age_mean = np.mean(soi.iloc[args.train_idx]['Age'])
    args.age_std = np.std(soi.iloc[args.train_idx]['Age'])  
    
    """Making Model"""
    FeaturesNet = m.GenerativeFeaturesNet(in_features=3)
    FeaturesNet = torch.nn.DataParallel(FeaturesNet)
    FeaturesNet.cuda()

    optimizer = torch.optim.Adam(FeaturesNet.parameters(),
                                 lr= args.lr)

    """Make PATH name"""
    name = 'cd'
    arguments = ('_seed'+str(args.seed)+
                 '_dice'+str(args.use_dice)+
                 '_ver_'+args.ver)

    PATH = args.save_dir+name+arguments+'/'

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        raise Exception('Path exists :(')

    """Train"""
    best = 1e8
    train_loss = []
    train_loss1 = []
    train_loss2 = []
    
    for epoch in tqdm(range(args.epochs)):
        mode = 'train'
        stat = run(loader= [dataset, train_sampler],
                   template= template,
                   models= [FeaturesNet],
                   optimizer= optimizer,
                   epoch= epoch,
                   mode= mode,
                   args= args,
                   PATH= PATH,
                   subsample_loader= len(dataset))

        train_loss += [stat[0]]
                                                 
        print('B.Sup Epoch %d' % (epoch))
        print('[Train] loss: %.3f'
              % (train_loss[-1]))

        train_summary = np.vstack((train_loss))

        """Save model"""
        # Using args.save_last always save the last epoch
        # Otherwise only save the loss improves

        state = {'epoch': epoch,
                 'args': args,
                 'FeaturesNet': FeaturesNet.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                 'train_summary': train_summary}
        
        if train_loss[-1]<best:
            best = train_loss[-1]
            torch.save(state, PATH+'best_model.pth.tar')

        if (epoch+1)%50 == 0:
            torch.save(state, PATH+'epoch{}_model.pth.tar'.format(epoch))