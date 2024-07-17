# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:16:14 2024

@author: haota
"""

import torch;
import numpy as np;

class predict_fns(object):
    
    def __init__(self, h, V, ne, nbasis, nframe, device):
        
        self.V = V;
        self.h = h;
        self.H = h+V;
        self.ne = ne;
        self.nframe = nframe;
        self.nbasis = nbasis;
        self.epsilon, phi = np.linalg.eigh(self.H.tolist());
        self.epsilon = torch.tensor(self.epsilon, dtype=torch.float32, device=device);
        phi = torch.tensor(phi, dtype=torch.float32, device=device);
        self.loss = torch.nn.MSELoss();
        self.co = phi[:, :, :ne];
        self.cv = phi[:, :, ne:];
        self.phi = phi;
        self.epsilon_ij = (self.epsilon[:, :ne, None] - self.epsilon[:, None, ne:])**-1;
        self.device = device;
    
    def E(self, Enn):
        
        Ehat = 2*torch.sum(self.epsilon[:, :self.ne], axis=1) + Enn;

        return Ehat;
    
    def O(self, O_mats):
        
        Ohat = 2 * torch.einsum('jui,juv,jvi->j', [self.co, O_mats, self.co])
        
        return Ohat;
    
    def C(self, C_mats):
        
        Chat = 2 * torch.einsum('jui,jsuv,jvi->js', [self.co, C_mats, self.co])
        
        return Chat;
        
    def B(self, B_mats):
        
        P = torch.einsum('ijk,ilk->ijl', [self.co, self.co]);
        
        Bhat = 4*torch.einsum('ijk,iukl,ilm,ivmj->iuv',
                            [P, B_mats, P, B_mats]);
        mask = 1 - torch.eye(len(Bhat[0])).to(self.device)[None,:,:];
        Bhat *= mask;

        return Bhat;

    def Eg(self, Gmat):
        
        Eg0 = self.epsilon[:, self.ne] - self.epsilon[:, self.ne-1];
        Eghat = Eg0 * (1 + Gmat.detach()[:,0]) + Gmat.detach()[:,1];

        return Eghat;
    
    def alpha(self, r_mats, T_mats, Gmat):
        
        r_all = torch.einsum('umi, xumn, unj -> uxij', [self.phi, r_mats, self.phi]);
        rij = r_all[:, :, :self.ne, self.ne:];

        epsilon_ij_G = (self.epsilon_ij**-1 * (1 + Gmat[:,0:1,None].detach()) - Gmat[:,1:2,None].detach())**-1;
        alpha_0 = - 4*torch.einsum('uxij, uyij, uij -> uxy', [rij, rij, epsilon_ij_G]);
        denominator = torch.linalg.inv(torch.eye(3).to(self.device)+torch.matmul(alpha_0,T_mats));
        alpha_hat = torch.matmul(denominator, alpha_0);

        return alpha_hat;

