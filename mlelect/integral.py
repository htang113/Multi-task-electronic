# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:44:30 2023

@author: 17000
"""

import json;
import torch;
from scipy.interpolate import lagrange;
import numpy as np;
import scipy;

class integrate():

    def __init__(self, device):

        with open('script/orbitals.json','r') as file:
            self.orbs_elements = json.load(file);
        self.device = device;
        self.zeta = {};
        self.alpha = {};
        self.c = {};
        self.Ng = {};
        self.Nb = {};
        
        for key in self.orbs_elements:
            self.zeta[key] = [];
            self.alpha[key] = [];
            self.c[key] = [];
            self.Ng[key] = 0;
            self.Nb[key] = 0;
            
            for u in self.orbs_elements[key]:
                self.zeta[key] += u['zeta'];
                self.alpha[key] += u['alpha'];
                self.c[key].append(u['c']);
                self.Ng[key] += len(u['c']);
                self.Nb[key] += 1;
        
        xlist = np.linspace(-3,3,7);
        matrix = [];
        for i in range(7):
            ylist = np.zeros(7);
            ylist[i] = 1;
            res = np.flip(lagrange(xlist, ylist).coef);
            res = res.tolist()+[0]*(7-len(res));
            matrix.append([res[2*n]*scipy.special.factorial2(max(2*n-1,0))/2**n*np.sqrt(np.pi) for n in range(4)]);
        self.mat = torch.tensor(matrix, dtype=torch.float).to(self.device);
                
    def Mat(self, z1, z2, x1, x2, k1, k2, polynomial):
        
        polynomial = torch.tensor(polynomial,
                                  dtype=torch.float)[None,None,None,:,None].to(self.device);
        xc = (z1[:,:,:,None]*x1+z2[:,:,:,None]*x2)/(z1+z2)[:,:,:,None];
        xc, r1, r2 = xc[:,:,:,:,None],(x1-xc)[:,:,:,:,None], (x2-xc)[:,:,:,:,None];
        exponent_term = torch.exp(-z1*z2/(z1+z2)*torch.sum((x1-x2)**2, axis=3));
        
        x = torch.linspace(-3,3,7)[None,None,None,None,:].to(self.device);
        integrant = (x+xc)**polynomial*(x-r1)**k1[:,:,:,:,None]*(x-r2)**k2[:,:,:,:,None];
        integrant = torch.einsum('ijklm,mn->ijkln', [integrant, self.mat]);
        
        divided = (z1+z2)[:,:,:,None,None]**torch.tensor([1/2+i for i in range(4)],
                                                         dtype=torch.float)[None,None,None,None,:].to(self.device);
        integrant /= divided;
        results = torch.sum(integrant, axis=4);
        results = torch.prod(results, axis=3)*exponent_term;
        
        return results;
    
    def calc_O(self, pos, atm, operator):
        
        polynomial = [operator.count('x'),
                      operator.count('y'),
                      operator.count('z')];
        
        angstron2Bohr = 1.88973;
        pos = pos.to(self.device)*angstron2Bohr;
        pos[:,:,[1,2,0]] = pos[:,:,[0,1,2]];
     
        mass = {'H':1.00794, 'C':12.011};
        ml = torch.tensor([mass[e1] for e1 in atm], dtype=torch.float).to(self.device);
        center = torch.einsum('uij,i->uj',[pos,ml])/torch.sum(ml);
        pos -= center[:,None,:];
        
        zeta, alpha, c, x = [], [], [], [];
        Nbasis, Ngaussian = 0,0;
        for i in range(len(atm)):
            atom = atm[i];
            zeta += self.zeta[atom];
            alpha += self.alpha[atom];
            x.append(pos[:,i:i+1,:].repeat([1,len(self.zeta[atom]),1]));
                
            c += self.c[atom];
            Ngaussian += self.Ng[atom];
            Nbasis += self.Nb[atom];
            
        x = torch.hstack(x).to(self.device);
        zeta = torch.tensor(zeta, dtype=torch.float).to(self.device);
        alpha = torch.tensor(alpha, dtype=torch.float).to(self.device);

        z1, z2 = zeta[None,:,None], zeta[None,None,:];
        k1, k2 = alpha[None,:,None,:], alpha[None,None,:,:];
        x1, x2 = x[:,:,None,:], x[:,None,:,:];
        
        Omat = self.Mat(z1, z2, x1, x2, k1, k2, polynomial);
        cmat = torch.zeros([Nbasis,Ngaussian], dtype=torch.float).to(self.device);
        
        c_index = 0;
        for i in range(len(c)):
            length = len(c[i]);
            cmat[i, c_index:c_index+length] = torch.tensor(c[i], dtype=torch.float).to(self.device);
            c_index += length;
            
        Omat = torch.einsum('ki,uij,lj->ukl',[cmat, Omat, cmat]);
        
        return Omat.tolist();
        
