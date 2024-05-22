# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:29:09 2023

@author: haota
"""


import time
import torch;
import json;
import numpy as np;
from pkgs.train import trainer;
from pkgs.dataframe import load_data;
from pkgs.model import V_theta;
from pkgs.integral import integrate;
from torch.optim.lr_scheduler import StepLR;

OPS = ['x', 'y', 'z', 'xx', 'yy', 'zz', 'xz', 'yz', 'xz']
device = 'cuda:0';
molecule_list = ['ethane'];

data, labels, obs_mat = load_data(molecule_list, device, load_obs_mat=False, 
                         ind_list=[0,1], op_names=OPS);
integrator = integrate(device);

pos = data[0]['pos'];
atm = data[0]['elements'];

S = torch.tensor(integrator.calc_S(pos,atm, ngrid=160)).to(device);

Sref = labels[0]['S'];

print(torch.mean((Sref-S)**2)/torch.mean(S**2));


