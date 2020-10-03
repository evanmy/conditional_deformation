import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

class AffineField:
    def __init__(self, batch_size, dim1, dim2, dim3, use_cuda=True, global_scale=False):
        self.batch_size = batch_size
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.use_cuda = use_cuda
        self.global_scale = global_scale 
        
    def normal_grid(self):
        lin1 = torch.linspace(-1, 1, self.dim1)
        lin2 = torch.linspace(-1, 1, self.dim2)
        lin3 = torch.linspace(-1, 1, self.dim3)
        mesh1, mesh2, mesh3 = torch.meshgrid(lin1, lin2, lin3)
        mesh1 = mesh1.unsqueeze(3)
        mesh2 = mesh2.unsqueeze(3)
        mesh3 = mesh3.unsqueeze(3)
        ones = torch.ones_like(mesh1)
        grid = torch.cat((mesh3, mesh2, mesh1, ones),3)
        grid = grid.unsqueeze(0)
        grid = grid.repeat(self.batch_size,1,1,1,1)
        grid = grid.float()
        if self.use_cuda:
            grid = grid.cuda()            
        grid = grid.view(-1, int(self.dim1*self.dim2*self.dim3), 4)
        return grid
    
    def deformed_grid(self, theta):        
        
        '''Create Matrix'''
        Mx = np.array([[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]])
        Mx = torch.from_numpy(Mx).repeat(self.batch_size,1,1).float()
        if self.use_cuda:
            Mx = Mx.cuda()
            
        My = np.array([[[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,1]]])
        My = torch.from_numpy(My).repeat(self.batch_size,1,1).float()
        if self.use_cuda:
            My = My.cuda()
            
        Mz = np.array([[[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]]])
        Mz = torch.from_numpy(Mz).repeat(self.batch_size,1,1).float()
        if self.use_cuda:
            Mz = Mz.cuda()
            
        '''Fill-in the Matrix'''
        # Rotation in x-axis
        Mx[:,1,1] = theta[:,0].cos()
        Mx[:,1,2] = -theta[:,0].sin()
        Mx[:,2,1] = theta[:,0].sin()
        Mx[:,2,2] = theta[:,0].cos()

        #Rotation in y-axis
        My[:,0,0] = theta[:,1].cos()
        My[:,0,2] = theta[:,1].sin()
        My[:,2,0] = -theta[:,1].sin()
        My[:,2,2] = theta[:,1].cos()

        #Rotation in z-axis
        Mz[:,0,0] = theta[:,2].cos()
        Mz[:,0,1] = -theta[:,2].sin()
        Mz[:,1,0] = theta[:,2].sin()
        Mz[:,1,1] = theta[:,2].cos()
          
        #Translation
        Mt = np.array([[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]])            
        Mt = torch.from_numpy(Mt).repeat(self.batch_size,1,1).float()
        if self.use_cuda:
            Mt = Mt.cuda()

        Mt[:,0,3] = theta[:,3]
        Mt[:,1,3] = theta[:,4]
        Mt[:,2,3] = theta[:,5]
        _M = torch.bmm(Mz, torch.bmm(My, torch.bmm(Mx,Mt)) )            

        #Fill in Scale
        Ms = np.array([[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]])
        Ms = torch.from_numpy(Ms).repeat(self.batch_size,1,1).float()    
        if self.use_cuda:
            Ms = Ms.cuda()
        
        if self.global_scale:
            Ms[:,0,0] = theta[:,6]**2
            Ms[:,1,1] = theta[:,6]**2
            Ms[:,2,2] = theta[:,6]**2
        else:
            Ms[:,0,0] = theta[:,6]**2
            Ms[:,1,1] = theta[:,7]**2
            Ms[:,2,2] = theta[:,8]**2
            
        _M = torch.bmm(Ms, _M)
        
        M = _M[:,:-1,:]
        M = torch.transpose(M, 1, 2)

        grid = self.normal_grid()
        phi = torch.bmm(grid, M)
        phi = phi.view(-1, self.dim1, self.dim2, self.dim3, 3)
        
        return phi
    
    def reflect(self, axis):        
        
        if axis==1:
            Mr = np.array([[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]])
            Mr = torch.from_numpy(Mr).repeat(self.batch_size,1,1).float()
            if self.use_cuda:
                Mr = Mr.cuda()
        elif axis==2:
            Mr = np.array([[[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]])
            Mr = torch.from_numpy(Mr).repeat(self.batch_size,1,1).float()
            if self.use_cuda:
                Mr = Mr.cuda()
        elif axis==3:
            Mr = np.array([[[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]]])
            Mr = torch.from_numpy(Mr).repeat(self.batch_size,1,1).float()
            if self.use_cuda:
                Mr = Mr.cuda()
        else:
            raise Exception('Not implemented')
        
        M = Mr[:,:-1,:]
        M = torch.transpose(M, 1, 2)

        grid = self.normal_grid()
        phi = torch.bmm(grid, M)
        phi = phi.view(-1, self.dim1, self.dim2, self.dim3, 3)
        
        return phi