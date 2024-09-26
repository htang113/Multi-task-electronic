# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:16:14 2024

@author: haota
"""

import torch;
import numpy as np;

class Losses(object):
    
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

    def V_loss(self):

        return torch.mean(self.V**2);
    
    def E_loss(self, E_labels):
        
        LE_grad, LE_out = 0, 0;
        for i,n in enumerate(self.ne):
            Ehat = 2*torch.sum(self.eo[i]);
            co = self.co[i];

            grad = 2*(Ehat-E_labels[i])*(2*torch.einsum('ij,kj,ik->',
                                        [co, co, self.H[i]]));  # energy term loss function for gradient
            
            LE_grad += grad;
            LE_out += (Ehat - E_labels[i])**2;
        
        return LE_grad/len(self.ne), LE_out/len(self.ne);
    
    def O_loss(self, O_labels, O_mats):

        LO_grad, LO_out = 0, 0;
        for i,n in enumerate(self.ne):

            Ohat = 2 * torch.einsum('ui,uv,vi->', [self.co[i], O_mats[i], self.co[i]]);
            O_term = (Ohat - O_labels[i]) * \
                     torch.einsum('uk,uv,vi->ik', [self.cv[i], O_mats[i], self.co[i]]) * \
                     torch.einsum('ui,uv,vk->ik', [self.co[i], self.H[i], self.cv[i]]);
            LO_grad += torch.sum(self.epsilon_ij[i] * O_term) * 2;
            LO_out += (Ohat - O_labels[i])**2/2;
        
        return LO_grad/len(self.ne), LO_out/len(self.ne);
    
    def C_loss(self, C_labels, C_mats):
        
        LC_grad, LC_out = 0, 0;
        for i,n in enumerate(self.ne):
            Chat = 2 * torch.einsum('ui,suv,vi->s', [self.co[i], C_mats[i], self.co[i]]);
            C_term = (Chat - C_labels[i])[:,None,None] * \
                     torch.einsum('uk,suv,vi->sik', [self.cv[i], C_mats[i], self.co[i]]) * \
                     torch.einsum('ui,uv,vk->ik', [self.co[i], self.H[i], self.cv[i]])[None,:,:];
            LC_grad += torch.mean(torch.einsum('ik,sik->s',self.epsilon_ij[i], C_term)) * 2;
            LC_out += torch.mean((Chat - C_labels[i])**2/2);

        return LC_grad/len(self.ne), LC_out/len(self.ne);
        
    def B_loss(self, B_labels, B_mats):
        
        LB_grad, LB_out = 0, 0;
        for i,n in enumerate(self.ne):
            P = torch.einsum('jk,lk->jl', [self.co[i], self.co[i]]);
            Bhat = 4*torch.einsum('jk,ukl,lm,vmj->uv', [P, B_mats[i], P, B_mats[i]]);
            mask = (B_labels[i] != 0);
            Bhat *= mask;
            term1 = 8*(Bhat-B_labels[i]);
            term2 = torch.einsum('mj,mn,ni,ij->ij', [self.cv[i], self.H[i], self.co[i], self.epsilon_ij[i]]);
            term31 = torch.einsum('mi,amn,nw,bwq,qj->abij', [self.co[i], B_mats[i], P, B_mats[i], self.cv[i]]);
            term32 = torch.einsum('mj,amn,nw,bwq,qi->abij', [self.cv[i], B_mats[i], P, B_mats[i], self.co[i]]);
            LB_grad += torch.mean(term1 * torch.einsum('ij,abij->ab', [term2, term31+term32]));
            LB_out += self.loss(Bhat, B_labels[i]);
        
        return LB_grad/len(self.ne), LB_out/len(self.ne);

    def Eg_loss(self, Eg_labels):

        Eg_grad, Eg_out = 0, 0;
        for i,n in enumerate(self.ne):
            Eg0 = self.epsilon[i, n] - self.epsilon[i, n-1];
            Eghat = Eg0 * (1 + self.G[i,0].detach()) + self.G[i,1].detach();
            grad_factor = 2*(Eghat - Eg_labels[i]);
            grad_term_1 = self.G[i,1] + (1 + self.G[i,0]) * \
                (torch.einsum('i,k,ik->',[self.cv[i][:,0], self.cv[i][:,0], self.H[i]]) - \
                 torch.einsum('i,k,ik->',[self.co[i][:,-1], self.co[i][:,-1], self.H[i]]));
            Eg_grad += grad_factor * grad_term_1;
            Eg_out += (Eghat - Eg_labels[i])**2;
        
        return Eg_grad/len(self.ne), Eg_out/len(self.ne);
    
    def polar_loss(self, alpha_labels, r_mats, smear=5E-3):
        
        T_mats = self.T;
        Gmat = self.G;

        polar_grad, polar_out = 0, 0;
        for i,n in enumerate(self.ne):

            r_all = torch.einsum('mi, xmn, nj -> xij', 
                                 [self.phi[i], r_mats[i], self.phi[i]]);
            rij = r_all[:, :n, n:];
            rik = r_all[:, :n, :];
            rjk = r_all[:, n:, :];
        
            V_all = torch.einsum('mi, mn, nj -> ij', 
                                 [self.phi[i], self.H[i], self.phi[i]]);
            Vij = V_all[:n, n:];
            Vik = V_all[:n, :];
            Vjk = V_all[n:, :];
            Vkk = torch.diagonal(V_all, dim1=0, dim2=1);
            V_diff = Vkk[:n,None] - Vkk[None,n:];
            epsilon_ij_G = (self.epsilon_ij[i]**-1 * (1 + Gmat[i,0:1,None].detach()) \
                            - Gmat[i,1:2,None].detach())**-1;

            alpha_0 = - 4*torch.einsum('xij, yij, ij -> xy', [rij, rij, epsilon_ij_G]);

            denominator = torch.linalg.inv(torch.eye(3).to(self.device)+torch.matmul(alpha_0,T_mats[i]));
            alpha_hat = torch.matmul(denominator, alpha_0);

            grad_term_1 = torch.einsum('xij, yij, ij, ij -> xy', 
                        [rij,rij,epsilon_ij_G**2, V_diff*(1 + Gmat[i,0:1,None])-Gmat[i,1:2,None]]);

            grad_term_2 = -2*torch.einsum('xij, yik, jk, ij, jk -> xy', 
                        [rij, rik, Vjk, epsilon_ij_G, self.epsilon_jk[i]]);
            grad_term_3 = -2*torch.einsum('xij, yjk, ik, ij, ik -> xy', 
                        [rij, rjk, Vik, epsilon_ij_G, self.epsilon_ik[i]]);

            polar_grad_tmp = 2*(alpha_hat.detach() - alpha_labels[i]) * (alpha_hat + \
                        torch.matmul(torch.matmul(denominator.detach(), grad_term_1 + grad_term_2 + grad_term_3),
                                    torch.eye(3).to(self.device)-torch.matmul(T_mats, alpha_hat).detach()));

            polar_grad += torch.mean(polar_grad_tmp);
            polar_out  += self.loss(alpha_hat, alpha_labels[i]);
        
        return polar_grad/len(self.ne), polar_out/len(self.ne);
