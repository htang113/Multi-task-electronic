# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:42:41 2023

@author: 17000
"""

from model.model_cls import V_theta
from basis.Irreps import Irreps_build
import numpy as np
import torch
from model.sample_minibatch import sampler;
from model.tomat import to_mat;
from torch.nn.parallel import DistributedDataParallel as DDP;
from train.loss_fns import Losses;
import os;

class trainer():

    def __init__(self, device, data_in, labels, op_matrices=[]) -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.sampler = sampler(data_in, labels, device);
        self.n_molecules = len(data_in);
        self.op_matrices = op_matrices;
    
    def build_irreps(self, element_list = ['H','C','N','O','F']):
        
        self.irreps = Irreps_build(element_list);
        self.irreps.generate_irreps();

        return None;

    def build_model(self, scaling = {'V':0.2, 'T': 0.01}):
        
        self.scaling = scaling;
        self.model = V_theta(self.device, self.irreps).to(self.device);
        self.transformer = to_mat(self.device, self.irreps);

    def build_ddp_model(self, scaling = {'V':0.2, 'T': 0.01}):
            
        self.scaling = scaling;
        self.model = DDP(V_theta(self.device,self.irreps).to(self.device), device_ids=[self.device]);
        self.transformer = to_mat(self.device, self.irreps);

    def build_optimizer(self, lr=10**-3):

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr);

    def build_charge_matrices(self, data_in):

        nbasis = self.irreps.get_nbasis();
        self.charge_matrices = [];
        for i in range(self.n_molecules):
            elements = data_in[i]['elements'];
            orbitals_list = [nbasis[u] for u in elements];
            map1 = [sum(orbitals_list[:j]) for j in range(len(elements)+1)];
            mati = [];
            for j in range(len(elements)):
                Sm = torch.zeros_like(data_in[i]['Smhalf']);
                Sm[map1[j]:map1[j+1],:] = \
                    data_in[i]['Smhalf'][map1[j]:map1[j+1],:];
                S = torch.matmul(torch.matmul(data_in[i]['Smhalf'], Sm),
                                 data_in[i]['S']);
                mati.append(S);
            self.charge_matrices.append(torch.stack(mati).to(self.device));

    def load(self, filename):

        if(os.path.exists(filename)):
            try:
                self.model.load_state_dict(torch.load(filename));
            except:
                try:
                    res = torch.load(filename);
                    for key in list(res.keys()):
                        res[key[7:]] = res[key];
                        del res[key];
                    self.model.load_state_dict(res);
                except:
                    res = torch.load(filename);
                    for key in list(res.keys()):
                        res['module.'+key] = res[key];
                        del res[key];
                    self.model.load_state_dict(res);

    def save(self, filename):
        
        torch.save(self.model.state_dict(), filename);

    def inference(self, minibatch):

        V_raw = self.model(minibatch);

        V, T, G = self.transformer.raw_to_mat(V_raw,minibatch);

        V *= self.scaling['V'];
        T *= self.scaling['T'];

        return V, T, G;
    
    def build_loss(self, loss_calculator, labels, ind, op_names):

        L_grads = {};
        L_ave = np.zeros(len(op_names));

        for i, op_name in enumerate(op_names):
            
            if(op_name == 'V'):
        
                L_grads[op_name] = loss_calculator.V_loss();
                L_ave[i] += L_grads[op_name];
                
            elif(op_name == 'E'):
                
                LE, LE_out = loss_calculator.E_loss(labels['Ee']);
                L_grads[op_name] = LE;
                L_ave[i] += LE_out;
            
            elif(op_name == 'atomic_charge'):
                
                C = labels['atomic_charge'];
                C_mat = [self.charge_matrices[k1] for k1 in ind];
                LC, LC_out = loss_calculator.C_loss(C, C_mat);
                L_grads[op_name] = LC;
                L_ave[i] += LC_out;
            
            elif(op_name == 'E_gap'):
                
                Egap = labels['E_gap'];
                Lgap_grad, Lgap_out = loss_calculator.Eg_loss(Egap);
                
                L_grads[op_name] = Lgap_grad;
                L_ave[i] += Lgap_out;
                
            elif(op_name == 'bond_order'):
                
                B = labels['B'];
                B_mat = [self.charge_matrices[k1] for k1 in ind];
                LB, LB_out = loss_calculator.B_loss(B, B_mat);
                L_grads[op_name] = LB;
                L_ave[i] += LB_out;
            
            elif(op_name == 'alpha'):
                
                alpha = labels['alpha'];
                r_mats = [torch.stack([self.op_matrices[ii][key] for key in ['x','y','z']]) \
                           for ii in ind];
                Lalpha_grad, Lalpha_out = loss_calculator.polar_loss(alpha, r_mats);

                L_grads[op_name] = Lalpha_grad;
                L_ave[i] += Lalpha_out;

            else:
                
                O = labels[op_name]
                O_mat = [self.op_matrices[k1][op_name] for k1 in ind];
                LO, LO_out = loss_calculator.O_loss(O, O_mat);
                L_grads[op_name] = LO;
                L_ave[i] += LO_out;
        
        regularization = sum([p.square().sum() for p in self.model.parameters()])/1E10;
        L = sum([op_names[key]*L_grads[key] \
                    for key in op_names])/len(ind) + regularization;

        return L, L_ave;

    def train(self, steps=10, batch_size = 50,
                    op_names=[]) -> float:

        # Train the model using given data points
        # M: number of data points, N: number of atoms, B: number of basis
        # pos_list: MxNx3 list of coordinates of input configurations
        # elements_list: MxN list of atomic species, 'C' or 'H'
        # E_list: M list of energy, unit Hartree
        # S_list: MxBxB list of overlap matrix <phi_i|phi_j>
        # N_list: MxBxB list of density matrix <phi_i|N|phi_j>
        # steps: steps to train using this dataset.
        # This method implements gradient descend to the contained model
        # and return the average loss
        
        operators_electric = [key for key in list(op_names.keys()) \
                              if key in ['x','y','z','xx','yy',
                                         'zz','xy','xz','yz']];
        L_ave = np.zeros(len(op_names));
        N_batchs = int(round(self.n_molecules/batch_size));

        for _ in range(steps):  # outer loop of training steps
            
            ########### forward calculations ################
            # apply the NN-model to get K-S potential

            for batch_ind in range(N_batchs):  # inner loop of batch size

                self.optim.zero_grad()  # clear gradient

                minibatch, labels, ind = self.sampler.sample(batch_ind = batch_ind, batch_size=batch_size,
                                                        irreps=self.irreps, op_names=operators_electric);
                
                V, T, G = self.inference(minibatch);
                
                # number of occupied orbitals
                h = minibatch['h'];
                ne = minibatch['ne'];
                norbs = minibatch['norbs'];
                loss_calculator = Losses(h, V, T, G, ne, norbs, self.device);
                
                loss_grad, loss_out = self.build_loss(loss_calculator, labels, ind, op_names);

                loss_grad.backward()  # calculate the gradient
                self.optim.step()  # implement gradient descend

                L_ave += loss_out;

        return L_ave/steps/N_batchs;


