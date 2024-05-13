# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:16:14 2024

@author: haota
"""

import torch;
import numpy as np;

class Losses(object):
    
    def __init__(self, h, V, ne, nbasis, nframe, device, smear = 5E-3):
        
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
        epsilon_jk = self.epsilon[:, self.ne:, None] - self.epsilon[:, None, :];
        self.epsilon_jk = epsilon_jk/(epsilon_jk**2 + smear**2);

        epsilon_ik = self.epsilon[:, :self.ne, None] - self.epsilon[:, None, :];
        self.epsilon_ik = epsilon_ik/(epsilon_ik**2 + smear**2);

    def V_loss(self):
        
        return torch.mean(self.V**2);
    
    def E_loss(self, E_labels):
        
        Ehat = 2*torch.sum(self.epsilon[:, :self.ne], axis=1);
        co = self.co;
        LE_grad = 2*(Ehat-E_labels)*(2*torch.einsum('uij,ukj,uik->u',
                                      [co, co, self.H]));  # energy term loss function for gradient
        
        LE_grad = torch.mean(LE_grad);
        LE_out = self.loss(Ehat,E_labels);
        
        return LE_grad, LE_out;
    
    def O_loss(self, O_labels, O_mats):
        
        Ohat = 2 * torch.einsum('jui,juv,jvi->j', [self.co, O_mats, self.co])
        O_term = (Ohat - O_labels)[:,None,None] * \
                 torch.einsum('juk,juv,jvi->jik', [self.cv, O_mats, self.co]) * \
                 torch.einsum('jui,juv,jvk->jik', [self.co, self.V, self.cv])
        LO_grad = torch.mean(torch.einsum('jik,jik->j',self.epsilon_ij, O_term) * 2);
            
        LO_out = torch.mean((Ohat - O_labels)**2/2);
        
        return LO_grad, LO_out;
    
    def C_loss(self, C_labels, C_mats):
        
        Chat = 2 * torch.einsum('jui,jsuv,jvi->js', [self.co, C_mats, self.co])
        C_term = (Chat - C_labels)[:,:,None,None] * \
                 torch.einsum('juk,jsuv,jvi->jsik', [self.cv, C_mats, self.co]) * \
                 torch.einsum('jui,juv,jvk->jik', [self.co, self.V, self.cv])[:,None,:,:]
        LC_grad = torch.mean(torch.einsum('jik,jsik->js',self.epsilon_ij, C_term)) * 2
        LC_out  = torch.mean((Chat - C_labels)**2/2);
        
        return LC_grad, LC_out;
        
    def B_loss(self, B_labels, B_mats):
        
        P = torch.einsum('ijk,ilk->ijl', [self.co, self.co]);
        
        Bhat = 4*torch.einsum('ijk,iukl,ilm,ivmj->iuv',
                            [P, B_mats, P, B_mats]);

        mask = (B_labels != 0)
        Bhat *= mask;

        term1 = 8*(Bhat-B_labels);
        
        term2 = torch.einsum('umj,umn,uni,uij->uij', 
                             [self.cv, self.V, self.co, self.epsilon_ij]);
        
        term31 = torch.einsum('umi,uamn,unw,ubwq,uqj->uabij', 
                              [self.co, B_mats, P, B_mats, self.cv]);
        
        term32 = torch.einsum('umj,uamn,unw,ubwq,uqi->uabij',
                              [self.cv, B_mats, P, B_mats, self.co]);
        
        LB_grad = term1*torch.einsum('uij,uabij->uab', 
                                     [term2, term31+term32]);
        LB_grad = torch.mean(LB_grad);
        LB_out = self.loss(Bhat, B_labels);
        
        return LB_grad, LB_out;

    def Eg_loss(self, Eg_labels, Gmat, C_mats):

        valence = self.co[:,:,-1];
        conduction = self.cv[:,:,0];
        Eg0 = self.epsilon[:, self.ne] - self.epsilon[:, self.ne-1];
        Eghat = Eg0 * (1 + Gmat.detach()[:,0]) + Gmat.detach()[:,1];

        grad_factor = 2*(Eghat - Eg_labels);
        grad_term_1 = Gmat[:,1] + (1 + Gmat[:,0]) * (torch.einsum('ui,uk,uik->u',[conduction, conduction, self.H]) - \
             torch.einsum('ui,uk,uik->u',[valence, valence, self.H]));

        Eg_grad = torch.mean(grad_factor * grad_term_1);
        Eg_out = self.loss(Eghat, Eg_labels);
        
        return Eg_grad, Eg_out;
    
    def polar_loss(self, alpha_labels, r_mats, T_mats, Gmat, smear=5E-3):
        
        r_all = torch.einsum('umi, xumn, unj -> uxij', [self.phi, r_mats, self.phi]);
        rij = r_all[:, :, :self.ne, self.ne:];
        rik = r_all[:, :, :self.ne, :];
        rjk = r_all[:, :, self.ne:, :];
        
        V_all = torch.einsum('umi, umn, unj -> uij', [self.phi, self.H, self.phi]);
        Vij = V_all[:, :self.ne, self.ne:];
        Vik = V_all[:, :self.ne, :];
        Vjk = V_all[:, self.ne:, :];
        Vkk = torch.diagonal(V_all, dim1=1, dim2=2);
        V_diff = Vkk[:,:self.ne,None] - Vkk[:,None,self.ne:];
        epsilon_ij_G = (self.epsilon_ij**-1 * (1 + Gmat[:,0:1,None].detach()) - Gmat[:,1:2,None].detach())**-1;

        alpha_0 = - 4*torch.einsum('uxij, uyij, uij -> uxy', [rij, rij, epsilon_ij_G]);
        denominator = torch.linalg.inv(torch.eye(3).to(self.device)+torch.matmul(alpha_0,T_mats));
        alpha_hat = torch.matmul(denominator, alpha_0);

        grad_term_1 = torch.einsum('uxij, uyij, uij, uij -> uxy',[rij,rij,epsilon_ij_G**2, V_diff*(1 + Gmat[:,0:1,None])-Gmat[:,1:2,None]]);
        grad_term_2 = -2*torch.einsum('uxij, uyik, ujk, uij, ujk -> uxy', [rij, rik, Vjk, epsilon_ij_G, self.epsilon_jk]);
        grad_term_3 = -2*torch.einsum('uxij, uyjk, uik, uij, uik -> uxy', [rij, rjk, Vik, epsilon_ij_G, self.epsilon_ik]);

        polar_grad = 2*(alpha_hat.detach() - alpha_labels) * (alpha_hat + \
                     torch.matmul(torch.matmul(denominator.detach(), grad_term_1 + grad_term_2 + grad_term_3),
                                  torch.eye(3).to(self.device)-torch.matmul(T_mats, alpha_hat).detach()));

        polar_grad = torch.mean(polar_grad);
        polar_out = self.loss(alpha_hat, alpha_labels);
        
        return polar_grad, polar_out;
