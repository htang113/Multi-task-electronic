#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:54:56 2023

@author: ubuntu
"""

from basis.integral import integrate
from model.model_cls import V_theta
from deploy.predictor import predict_fns
import numpy as np
import torch
from model.tomat import to_mat;
from train.train_cls import trainer;
import os;
import json;
import matplotlib;
import matplotlib.pyplot as plt;

class estimator_test(trainer):

    def __init__(self, device, data_in, labels, op_matrices=[], output_folder='output/test') -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        trainer.__init__(self, device, data_in, labels, op_matrices);

        if(not os.path.exists(output_folder)):
            os.mkdir(output_folder);


    def evaluate_properties(self, calc, ind, op_names):

        property_dic = {};

        for i, op_name in enumerate(op_names):
                
            if(op_name == 'E'):
                
                LE = calc.E();
                property_dic[op_name] = [float(Ei) for Ei in LE];
            
            elif(op_name == 'atomic_charge'):
                
                C_mat = [self.charge_matrices[k1] for k1 in ind];
                LC = calc.C(C_mat);
                property_dic[op_name] = [Ci.tolist() for Ci in LC];
            
            elif(op_name == 'E_gap'):
                
                LEg = calc.Eg();
                property_dic[op_name] = [float(Egi) for Egi in LEg];
                
            elif(op_name == 'bond_order'):
                
                B_mat = [self.charge_matrices[k1] for k1 in ind];
                LB = calc.B(B_mat);
                property_dic[op_name] = [Bi.tolist() for Bi in LB];
            
            elif(op_name == 'alpha'):
                
                r_mats = [torch.stack([self.op_matrices[ii][key] for key in ['x','y','z']]) \
                           for ii in ind];
                Lalpha = calc.alpha(r_mats);
                property_dic[op_name] = [alphai.tolist() for alphai in Lalpha];

            else:
                
                O_mat = [self.op_matrices[k1][op_name] for k1 in ind];
                LO = calc.O(O_mat);
                property_dic[op_name] = [float(Oi) for Oi in LO];

        return property_dic;

    def solve(self, batch_size=1, op_names=[]):

        operators_electric = [key for key in list(op_names.keys()) \
                        if key in ['x','y','z','xx','yy',
                                    'zz','xy','xz','yz']];
        N_batchs = int(round(self.n_molecules/batch_size));
        properties_dic = {};

        for batch_ind in range(N_batchs):
            minibatch, labels, ind = self.sampler.sample(batch_ind = batch_ind, batch_size=batch_size,
                                                    irreps=self.irreps, op_names=operators_electric);
            
            V, T, G = self.inference(minibatch);
            
            # number of occupied orbitals
            h = minibatch['h'];
            ne = minibatch['ne'];
            norbs = minibatch['norbs'];
            property_calc = predict_fns(h, V, T, G, ne, norbs, self.device);
            
            properties = self.evaluate_properties(property_calc, ind, op_names);
            for key in properties:
                if(key not in properties_dic):
                    properties_dic[key] = {'pred':[],'label':[]};
                properties_dic[key]['pred'] += properties[key];
                if(key=='bond_order'):
                    properties_dic[key]['label'] += [u.tolist() for u in labels['B']];
                elif(key == 'E'):
                    properties_dic[key]['label'] += labels['Ee'];
                elif(key in ['atomic_charge','alpha']):
                    properties_dic[key]['label'] += [u.tolist() for u in labels[key]];
                else:
                    properties_dic[key]['label'] += labels[key];

            print('complete ' + str(batch_ind) + ' of ' + str(N_batchs));

        return properties_dic;
