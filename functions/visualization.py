import torch
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.filters import gaussian
from skimage.transform import rotate

def overlay_regions(mri, 
                    regs,
                    image_idx, 
                    brain_oppacity=0.6, 
                    show_brain=True, 
                    ax=None,
                    title='Overlay', 
                    s=None):
    '''
    mri: [n_batch, 1, x_dim, y_dim, z_dim] torch tensor
    regs: [n_batch, n_labels, x_dim, y_dim, z_dim] torch tensor
    
    '''
    mri = mri.detach().cpu()
    regs = regs.detach().cpu()

    if mri.dtype is not torch.float32:
        mri = torch.from_numpy(mri).float()
    if regs.dtype is not torch.float32:
        regs = torch.from_numpy(regs).float()
    
    regs = torch.clamp(regs,0,1) 
    mri = torch.clamp(mri,0,1)   
    n_batch, n_labels, x_dim, y_dim, z_dim = regs.shape
    
    if s is None:
        _s = z_dim//2
    else:
        _s = s
    
    mri_slice = mri[image_idx, 0, :, :, _s]
#     mri_slice = rotate(mri_slice, angle=-90, mode='edge')
    mri_slice = np.swapaxes(mri_slice.float().numpy(),0,1)

    #Note the first and last colors are not used
    #First color is skip because is the background label
    #Last color is white, which might be confusing
    cmap = matplotlib.cm.get_cmap('tab20c')
#     cmap = matplotlib.cm.get_cmap('tab20')
    palette = ((1.0,1.0,1.0),) + cmap.colors[:n_labels-2] + ((1.0,1.0,1.0),)
    if ax is None:
        fig, ax = plt.subplots()
    else: 
        ax = ax
    for i in range(n_labels):   
        if i == 0:
            continue
        slice = regs[image_idx, i, :, :, _s].float().numpy()
#         slice = rotate(slice, angle=-90, mode='edge')
        slice = np.swapaxes(slice,0,1)
        alpha = np.expand_dims(slice,2)
        rbg = np.asarray(palette[i]).reshape(1,1,-3)    
        slice_rgb = rbg*np.tile(np.expand_dims(slice, 2),(1,1,3))
        slice_rgba = np.concatenate([slice_rgb, alpha], 2)  

        ax.imshow(slice_rgba)

    if show_brain:
        #Make brain transparent and solid background
        alpha_bckgrnd = (mri_slice<0.001)*1
        alpha_brain = (mri_slice>0.001)*brain_oppacity
        alpha = alpha_bckgrnd+alpha_brain
        alpha = gaussian(alpha, sigma=1.5) #For the edges to look nice
        alpha = np.expand_dims(alpha,2)

        mri_slice = np.tile(np.expand_dims(mri_slice,2),(1,1,3))
        mri_rbga = np.concatenate((mri_slice, alpha), 2)

        ax.imshow(mri_rbga)
          
    ax.axis('off')
    ax.set_title(title)

    
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

def view_results(target_brain, 
                 recon_brain, 
                 pred_regs,
                 target_regs,
                 epoch, 
                 suffix, 
                 show_brain,
                 image_idx=0,
                 PATH=None, 
                 show_image=False, 
                 s =None):
    if s is None:
        _s = target_brain.shape[4]//2
    else:
        _s = s
        
    # Cross-section
    _target_brain = target_brain
    recon_brain = recon_brain[image_idx,0,:,:,_s].data.cpu().numpy()
    target_brain = target_brain[image_idx,0,:,:,_s].data.cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=4)
    
    ax[0].set_title('Recon')
    recon_brain = np.clip(recon_brain, 0, 1)
#     image = rotate(recon_brain, angle=-90, mode='edge')
    image = np.swapaxes(recon_brain,0,1)
    ax[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    ax[0].axis('off')    
      
    ax[1].set_title('Target')
#     image = rotate(target_brain, angle=-90, mode='edge')
    image = np.swapaxes(target_brain,0,1)
    ax[1].imshow(image, cmap='gray', vmin=0, vmax=1)
    ax[1].axis('off')
    
    overlay_regions(_target_brain, 
                    pred_regs,
                    image_idx,
                    brain_oppacity=0.2,
                    show_brain=show_brain,
                    ax=ax[2], 
                    title='Predicted', 
                    s= _s)
    
    overlay_regions(_target_brain, 
                    target_regs,
                    image_idx,
                    brain_oppacity=0.2,
                    show_brain=show_brain,
                    ax=ax[3], 
                    title='Target', 
                    s= _s)    
    if PATH is not None:
        try:
            fig.set_size_inches(15, 5)
            fig.savefig(PATH+str(epoch)+suffix)
        except:
            fig.set_size_inches(15, 5)
            fig.savefig(PATH+str(epoch)+suffix+'.png')            
    elif show_image:
        plt.show()
        
    plt.close('all')
    
def view_results2(target_brain, 
                  recon_brain, 
                  pred_regs,
                  prior,
                  target_regs,
                  epoch, 
                  suffix, 
                  show_brain,
                  image_idx=0,
                  PATH=None, 
                  show_image=False):
    
    _s = target_brain.shape[4]//2
    
    # Cross-section
    _target_brain = target_brain
    recon_brain = recon_brain[image_idx,0,:,:,_s].data.cpu().numpy()
    target_brain = target_brain[image_idx,0,:,:,_s].data.cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=5)
    
    ax[0].set_title('Recon')
    recon_brain = np.clip(recon_brain, 0, 1)
    image = np.swapaxes(recon_brain,0,1)
    ax[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    ax[0].axis('off')    
      
    ax[1].set_title('Target')
    image = np.swapaxes(target_brain,0,1)
    ax[1].imshow(image, cmap='gray', vmin=0, vmax=1)
    ax[1].axis('off')
    
    overlay_regions(_target_brain, 
                    pred_regs,
                    image_idx,
                    brain_oppacity=0.2,
                    show_brain=show_brain,
                    ax=ax[2], title='Predicted')
    
    overlay_regions(_target_brain, 
                    prior,
                    image_idx,
                    brain_oppacity=0.2,
                    show_brain=show_brain,
                    ax=ax[3], title='Prior')
    
    overlay_regions(_target_brain, 
                    target_regs,
                    image_idx,
                    brain_oppacity=0.2,
                    show_brain=show_brain,
                    ax=ax[4], 
                    title='Target')    
    if PATH is not None:
        fig.set_size_inches(15, 5)
        fig.savefig(PATH+str(epoch)+suffix)
    elif show_image:
        plt.show()
        
    plt.close('all')
    
def view_prior(target_brain,
               prior, 
               deformed_prior, 
               deformed_grid,
               epoch, 
               suffix, 
               show_brain,
               image_idx=0,
               PATH=None, 
               show_image=False):
    
    _s = target_brain.shape[4]//2
    _target_brain = target_brain
    
    # Cross-section
    target_brain = target_brain[image_idx,0,:,:,_s].data.cpu().numpy()
    deformed_grid = deformed_grid[image_idx,0,:,:,_s].data.cpu().numpy()
    
    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=4)
    overlay_regions(_target_brain, 
                    prior,
                    image_idx,
                    brain_oppacity=0.2,
                    show_brain=show_brain,
                    ax=ax[0], title='Prior')
    
    ax[1].set_title('Target Brain')
#     image = rotate(target_brain, angle=-90, mode='edge')
    image = np.swapaxes(target_brain,0,1)
    ax[1].imshow(image, cmap='gray', vmin=0, vmax=1)
    ax[1].axis('off')    
    
    overlay_regions(_target_brain, 
                    deformed_prior,
                    image_idx,
                    brain_oppacity=0.2,
                    show_brain=show_brain,
                    ax=ax[2], 
                    title='Deformed Prior')    
    
    ax[3].set_title('Grid')
#     image = rotate(deformed_grid, -90, mode='edge')
    image = np.swapaxes(deformed_grid,0,1)
    ax[3].imshow(image, cmap='gray', vmin=0, vmax=1)
    ax[3].axis('off') 
      
    if PATH is not None:
        fig.set_size_inches(15, 5)
        fig.savefig(PATH+str(epoch)+suffix+'prior')
    elif show_image:
        plt.show()
        
    plt.close('all')