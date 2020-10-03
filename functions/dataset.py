import torch
import numpy as np
import pandas as pd
import nibabel as nib
import torch.utils.data as data
import torch.nn.functional as F
from functions import training_tools as tt
from functions import spatial_transformer as st

def get_affine_atlas(path):
    atlas = torch.from_numpy(np.load((path))['vol_data']).float()

    subset_regs = [[0,0],   #Background
                   [13,52], #Pallidum   
                   [18,54], #Amygdala
                   [11,50], #Caudate
                   [3,42],  #Cerebral Cortex
                   [17,53], #Hippocampus
                   [10,49], #Thalamus
                   [12,51], #Putamen
                   [2,41],  #Cerebral WM
                   [8,47],  #Cerebellum Cortex
                   [4,43],  #Lateral Ventricle
                   [7,46],  #Cerebellum WM
                   [16,16]] #Brain-Stem

    atlas_label = np.array([0,2,3,4,5,7,8,10,
                            11,12,13,14,15,16,
                            17,18,24,26,28,29,
                            30,31,41,42,43,44,
                            46,47,49,50,51,52,
                            53,54,58,60,62,63,
                            72,77,80,85])

    dim1, dim2, dim3, _= atlas.shape
    chs = 14
    one_hot = torch.zeros(1, chs, dim1, dim2, dim3)

    for i,s in enumerate(subset_regs):
        if not s[0]==s[1]:
            idx1 = np.argwhere(atlas_label == s[0]).flatten()[0]
            idx2 = np.argwhere(atlas_label == s[1]).flatten()[0]
            one_hot[0,i,:,:,:] = atlas[:,:,:,idx1] + atlas[:,:,:,idx2]
        else:
            idx = np.argwhere(atlas_label == s[0]).flatten()[0]
            one_hot[0,i,:,:,:] = atlas[:,:,:,idx]

    mask = one_hot.sum(1).squeeze()
    mask = torch.clamp(mask, 0, 1) 
    ones = torch.ones_like(mask)
    non_roi = ones-mask  
    one_hot[0,-1,:,:,:] = non_roi    
    assert one_hot.sum(1).sum() == dim1*dim2*dim3, 'One-hot encoding does not added up to 1'
    return one_hot

def get_nonlinear_atlas(path):
    atlas = torch.from_numpy(np.load((path))['vol_data']).float()
    a = atlas[:,:,:,:3].sum(-1, True) #combine skull with bck
    b = atlas[:,:,:,3:]               #combine soft tissue with bck
    atlas = torch.cat((a,b),-1) 
    
    subset_regs = [[0,0],   #Background
                   [13,52], #Pallidum   
                   [18,54], #Amygdala
                   [11,50], #Caudate
                   [3,42],  #Cerebral Cortex
                   [17,53], #Hippocampus
                   [10,49], #Thalamus
                   [12,51], #Putamen
                   [2,41],  #Cerebral WM
                   [8,47],  #Cerebellum Cortex
                   [4,43],  #Lateral Ventricle
                   [7,46],  #Cerebellum WM
                   [16,16]] #Brain-Stem

    atlas_label = np.array([0,16,24,8,47,15,
                            259,3,42,7,46,41,2,63,54,
                            18,53,44,17,31,5,85,60,28,
                            14,26,12,51,62,58,77,4,30,
                            43,52,11,49,13,50,10,80,72])

    dim1, dim2, dim3, _= atlas.shape
    chs = 14
    one_hot = torch.zeros(1, chs, dim1, dim2, dim3)

    for i,s in enumerate(subset_regs):
        if not s[0]==s[1]:
            idx1 = np.argwhere(atlas_label == s[0]).flatten()[0]
            idx2 = np.argwhere(atlas_label == s[1]).flatten()[0]
            one_hot[0,i,:,:,:] = atlas[:,:,:,idx1] + atlas[:,:,:,idx2]
        else:
            idx = np.argwhere(atlas_label == s[0]).flatten()[0]
            one_hot[0,i,:,:,:] = atlas[:,:,:,idx]

    mask = one_hot.sum(1).squeeze()
    mask = torch.clamp(mask, 0, 1) 
    ones = torch.ones_like(mask)
    non_roi = ones-mask  
    one_hot[0,-1,:,:,:] = non_roi    
    assert one_hot.sum(1).sum() == dim1*dim2*dim3, 'One-hot encoding does not added up to 1'
    return one_hot

def get_onehot13(asegs):
    subset_regs = [[0,0],
                   [13,52], 
                   [18,54], 
                   [11,50], 
                   [3,42], 
                   [17,53], 
                   [10,49],
                   [12,51], 
                   [2,41], 
                   [8,47], 
                   [4,43], 
                   [7,46],
                   [16,16]]

    dim1, dim2, dim3 = asegs.shape
    chs = 14
    one_hot = torch.zeros(1, chs, dim1, dim2, dim3)

    for i,s in enumerate(subset_regs):
        combined_vol = (asegs == s[0]) | (asegs == s[1]) 
        one_hot[:,i,:,:,:] = torch.from_numpy(combined_vol*1).float()

    mask = one_hot.sum(1).squeeze() 
    ones = torch.ones_like(mask)
    non_roi = ones-mask    
    one_hot[0,-1,:,:,:] = non_roi    

    assert one_hot.sum(1).sum() == dim1*dim2*dim3, 'One-hot encoding does not added up to 1'
    return one_hot

def get_subject(path,
                use_healthy= True):
    '''
    Return a pandas file with subject that meets the input criteria
    soi: subject of interest
    '''
    phen = pd.read_csv(path)
    phen['M/F']  = (phen['M/F'] == 'F')*1
    phen['CDR'] = phen['CDR'].fillna(0)
    phen['CDR'] = (phen['CDR']>0.0)*1
    CDR = phen['CDR']
        
    if use_healthy:
        idx = np.argwhere(CDR == 0).flatten()
    else:
        idx = np.arange(len(CDR))

    _soi = phen.iloc[idx]
    _soi = _soi.reset_index(drop=True)
        
    return _soi
    
class load_oasis_data(data.Dataset):
    def __init__(self, soi):
        super(load_oasis_data, self).__init__()
        """
        soi: pandas dataframe with subject of interet
        reg: region of the brain
        """
        self.soi = soi

    def __len__(self):
        return self.soi.shape[0]
    
    def __getitem__(self, index):
        
        if type(index)==np.ndarray or type(index)==torch.Tensor:
            assert len(index) == 1, 'Only minibatch of 1 supported'
            index = index[0]    
            
        id = self.soi.iloc[index]['ID']

        PATH = './files/'
        mri = np.load(PATH + 
                      id + 
                      '.npz')                
        mri = mri['vol_data']
        mri = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0)
        aseg = np.load(PATH + 
                       id + 
                      '_aseg.npz')
        
        aseg = aseg['vol_data'].astype('float32')
        onehot = get_onehot13(aseg)
        aseg = torch.from_numpy(aseg)
        
        mri = mri.float()
        phenotype = self.soi.iloc[[index]]
        return mri, aseg, onehot, phenotype
    
def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax