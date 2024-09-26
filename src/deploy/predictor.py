# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:16:14 2024

@author: haota
"""

import torch;
import numpy as np;

class predict_fns(object):
    
    def __init__(self, h, V, T, G, ne, norbs, device, smear = 5E-3):

        self.device = device;
        self.V = V;
        self.h = h;
        self.T = T;
        self.G = G;
        H = h+V;
        self.ne = ne;
        self.norbs = norbs;
        self.epsilon, phi = np.linalg.eigh(H.tolist());
        self.epsilon = torch.tensor(self.epsilon, dtype=torch.float32).to(device);
        phi = torch.tensor(phi, dtype=torch.float32).to(device);
        self.loss = torch.nn.MSELoss();

        self.co = [];
        self.cv = [];
        self.eo = [];
        self.ev = [];
        self.epsilon_ij = [];
        self.epsilon_jk = [];
        self.epsilon_ik = [];
        self.H = [];
        self.phi = [];

        for i, n in enumerate(ne):
            nb = norbs[i]

            self.eo.append(self.epsilon[i, :n]);
            self.co.append(phi[i, :nb, :n]);
            self.ev.append(self.epsilon[i, n:nb]);
            self.cv.append(phi[i, :nb, n:nb]);
            self.phi.append(phi[i, :nb, :nb]);
        
            ij = self.eo[i][:,None]-self.ev[i][None,:];
            self.epsilon_ij.append(ij**-1)
        
            epsilon_jk = self.ev[i][:, None] - self.epsilon[i][ None, :nb];
            self.epsilon_jk.append(epsilon_jk/(epsilon_jk**2 + smear**2));

            epsilon_ik = self.eo[i][ :, None] - self.epsilon[i][ None, :nb];
            self.epsilon_ik.append(epsilon_ik/(epsilon_ik**2 + smear**2));

            self.H.append(H[i,:nb,:nb]);
    
    def E(self):
        
        Ehat = [];
        for i,n in enumerate(self.ne):
            Ehat.append(2*torch.sum(self.eo[i]));
        
        return Ehat;
    
    def O(self, O_mats):

        Ohat = [];
        for i,n in enumerate(self.ne):

            Ohat.append(2 * torch.einsum('ui,uv,vi->', 
                        [self.co[i], O_mats[i], self.co[i]]));
        
        return Ohat;
    
    def C(self, C_mats):
        
        Chat = [];
        for i,n in enumerate(self.ne):
            Chat.append(2 * torch.einsum('ui,suv,vi->s', 
                    [self.co[i], C_mats[i], self.co[i]]));

        return Chat;
        
    def B(self, B_mats):
        
        Bhat = [];
        for i,n in enumerate(self.ne):
            P = torch.einsum('jk,lk->jl', [self.co[i], self.co[i]]);
            Bhati = 4*torch.einsum('jk,ukl,lm,vmj->uv', [P, B_mats[i], P, B_mats[i]]);
            mask = 1 - torch.eye(len(Bhati[0])).to(self.device);
            Bhati *= mask;
            Bhat.append(Bhati);
        
        return Bhat;

    def Eg(self):

        Eghat = [];
        for i,n in enumerate(self.ne):
            Eg0 = self.epsilon[i, n] - self.epsilon[i, n-1];
            Eghat.append(Eg0 * (1 + self.G[i,0].detach()) + self.G[i,1].detach());
        
        return Eghat;
    
    def alpha(self, r_mats):
        
        alpha_hat = [];

        T_mats = self.T;
        Gmat = self.G;
        
        for i,n in enumerate(self.ne):

            r_all = torch.einsum('mi, xmn, nj -> xij', 
                                 [self.phi[i], r_mats[i], self.phi[i]]);
            rij = r_all[:, :n, n:];

            epsilon_ij_G = (self.epsilon_ij[i]**-1 * (1 + Gmat[i,0:1,None].detach()) \
                            - Gmat[i,1:2,None].detach())**-1;

            alpha_0 = - 4*torch.einsum('xij, yij, ij -> xy', [rij, rij, epsilon_ij_G]);

            denominator = torch.linalg.inv(torch.eye(3).to(self.device)+torch.matmul(alpha_0,T_mats[i]));
            alpha_hat.append(torch.matmul(denominator, alpha_0));
        
        return alpha_hat;