# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:42:41 2023

@author: 17000
"""

from mtelect.integral import integrate
from mtelect.model import V_theta
import numpy as np
import torch
from mtelect.sample_minibatch import sampler;
from mtelect.tomat import to_mat;
from torch.nn.parallel import DistributedDataParallel as DDP;
from mtelect.loss_fns import Losses;

class trainer():

    def __init__(self, device, data_in, labels, op_matrices=[], 
                 filename='model.pt', lr=10**-3, scaling = 0.1) -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.lr = lr
        self.model = V_theta(device).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.filename = filename;
        self.sampler = sampler(data_in, labels, device);
        self.n_molecules = len(data_in);
        self.transformer = to_mat(device);
        self.scaling = scaling;
        
        self.op_matrices = op_matrices;
        
        self.charge_matrices = [];
        for i in range(self.n_molecules):
            elements = data_in[i]['elements'];
            orbitals_list = [9*(u=='C')+5 for u in elements];
            map1 = [sum(orbitals_list[:j]) for j in range(len(elements)+1)];
            mati = [];
            for j in range(len(elements)):
                Sm = torch.zeros_like(labels[i]['Smhalf']);
                Sm[:,map1[j]:map1[j+1],:] = \
                    labels[i]['Smhalf'][:,map1[j]:map1[j+1],:];
                S = torch.matmul(torch.matmul(labels[i]['Smhalf'], Sm),
                                 labels[i]['S']);
                mati.append(S[:,None,:,:]);
            self.charge_matrices.append(torch.hstack(mati));

    def load(self, filename):
        
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
        
        for _ in range(steps):  # outer loop of training steps

            ########### forward calculations ################
            # apply the NN-model to get K-S potential
            self.optim.zero_grad()  # clear gradient
            
            for i_m in range(self.n_molecules):

                minibatch, labels = self.sampler.sample(batch_size=batch_size, i_molecule=i_m,
                                                        op_names=operators_electric);

                h = labels['h'];

                # number of occupied orbitals
                ne = labels['ne'];

                nbasis = labels['norbs']
                nframe = labels['nframe']

                E = labels['Ee'];  # ccsdt total energy

                V_raw = self.model(minibatch);

                V, T, G = self.transformer.raw_to_mat(V_raw,minibatch,labels);
                V *= self.scaling['V'];
                T *= self.scaling['T'];
                
                loss_calculator = Losses(h, V, ne, nbasis, nframe, self.device);
                
                L_grads = {};
                
                for i, op_name in enumerate(op_names):
                    
                    if(op_name == 'V'):
                
                        L_grads[op_name] = loss_calculator.V_loss();
                        L_ave[i] += L_grads[op_name];
                        
                    elif(op_name == 'E'):
                        
                        LE, LE_out = loss_calculator.E_loss(E);
                        L_grads[op_name] = LE;
                        L_ave[i] += LE_out;
                    
                    elif(op_name == 'atomic_charge'):
                        
                        C = labels['atomic_charge'];
                        C_mat = self.charge_matrices[i_m];
                        LC, LC_out = loss_calculator.C_loss(C, C_mat);
                        L_grads[op_name] = LC;
                        L_ave[i] += LC_out;
                    
                    elif(op_name == 'E_gap'):
                        
                        Egap = labels['E_gap'];
                        Lgap_grad, Lgap_out = loss_calculator.Eg_loss(Egap, G, C_mat);
                        
                        L_grads[op_name] = Lgap_grad;
                        L_ave[i] += Lgap_out;
                        
                    elif(op_name == 'bond_order'):
                        
                        B = labels['B'];
                        B_mat = self.charge_matrices[i_m];
                        LB, LB_out = loss_calculator.B_loss(B, B_mat);
                        L_grads[op_name] = LB;
                        L_ave[i] += LB_out;
                    
                    elif(op_name == 'alpha'):
                        
                        alpha = labels['alpha'];
                        r_mats = torch.stack([self.op_matrices[i_m]['x'],
                                                self.op_matrices[i_m]['y'],
                                                self.op_matrices[i_m]['z']]);
                        Lalpha_grad, Lalpha_out = loss_calculator.polar_loss(alpha, r_mats, T, G);

                        L_grads[op_name] = Lalpha_grad;
                        L_ave[i] += Lalpha_out;
                        
                    else:
                        
                        O = labels[op_name]
                        O_mat = self.op_matrices[i_m][op_name];
                        LO, LO_out = loss_calculator.O_loss(O, O_mat);
                        L_grads[op_name] = LO;
                        L_ave[i] += LO_out;
                        
                L = sum([op_names[key]*L_grads[key] \
                         for key in op_names])/self.n_molecules;
        
                L.backward()  # calculate the gradient

            self.optim.step()  # implement gradient descend


        torch.save(self.model.state_dict(), self.filename);

        return L_ave/steps/self.n_molecules;


class trainer_ddp():

    def __init__(self, device, data_in, labels, 
                 op_matrices=[],filename='model.pt', lr=10**-3,
                 scaling=0.1) -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.lr = lr
        self.model = DDP(V_theta(device).to(device), device_ids=[device]);
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss();
        self.filename = filename;
        self.sampler = sampler(data_in, labels, device);
        self.n_molecules = len(data_in);
        self.transformer = to_mat(device);
        self.op_matrices = op_matrices;
        self.scaling = scaling;
        
        self.charge_matrices = [];
        for i in range(self.n_molecules):
            elements = data_in[i]['elements'];
            orbitals_list = [9*(u=='C')+5 for u in elements];
            map1 = [sum(orbitals_list[:j]) for j in range(len(elements)+1)];
            mati = [];
            for j in range(len(elements)):
                Sm = torch.zeros_like(labels[i]['Smhalf']);
                Sm[:,map1[j]:map1[j+1],:] = \
                    labels[i]['Smhalf'][:,map1[j]:map1[j+1],:];
                S = torch.matmul(torch.matmul(labels[i]['Smhalf'], Sm),
                                 labels[i]['S']);
                mati.append(S[:,None,:,:]);
            self.charge_matrices.append(torch.hstack(mati));
            
    def load(self, filename):
        
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
        
        for _ in range(steps):  # outer loop of training steps

            ########### forward calculations ################
            # apply the NN-model to get K-S potential
            self.optim.zero_grad()  # clear gradient
            
            for i_m in range(self.n_molecules):

                minibatch, labels = self.sampler.sample(batch_size=batch_size, i_molecule=i_m,
                                                        op_names=operators_electric);

                h = labels['h'];

                # number of occupied orbitals
                ne = labels['ne'];

                nbasis = labels['norbs']
                nframe = labels['nframe']

                E = labels['Ee'];  # ccsdt total energy

                V_raw = self.model(minibatch);

                V, T, G = self.transformer.raw_to_mat(V_raw,minibatch,labels);
                V *= self.scaling['V'];
                T *= self.scaling['T'];
                
                loss_calculator = Losses(h, V, ne, nbasis, nframe, self.device);
                
                L_grads = {};
                
                for i, op_name in enumerate(op_names):
                    
                    if(op_name == 'V'):
                
                        L_grads[op_name] = loss_calculator.V_loss();
                        L_ave[i] += L_grads[op_name];
                        
                    elif(op_name == 'E'):
                        
                        LE, LE_out = loss_calculator.E_loss(E);
                        L_grads[op_name] = LE;
                        L_ave[i] += LE_out;
                    
                    elif(op_name == 'atomic_charge'):
                        
                        C = labels['atomic_charge'];
                        C_mat = self.charge_matrices[i_m];
                        LC, LC_out = loss_calculator.C_loss(C, C_mat);
                        L_grads[op_name] = LC;
                        L_ave[i] += LC_out;
                    
                    elif(op_name == 'E_gap'):
                        
                        Egap = labels['E_gap'];
                        Lgap_grad, Lgap_out = loss_calculator.Eg_loss(Egap, G, C_mat);
                        
                        L_grads[op_name] = Lgap_grad;
                        L_ave[i] += Lgap_out;
                        
                    elif(op_name == 'bond_order'):
                        
                        B = labels['B'];
                        B_mat = self.charge_matrices[i_m];
                        LB, LB_out = loss_calculator.B_loss(B, B_mat);
                        L_grads[op_name] = LB;
                        L_ave[i] += LB_out;
                    
                    elif(op_name == 'alpha'):
                        
                        alpha = labels['alpha'];
                        r_mats = torch.stack([self.op_matrices[i_m]['x'],
                                                self.op_matrices[i_m]['y'],
                                                self.op_matrices[i_m]['z']]);
                        Lalpha_grad, Lalpha_out = loss_calculator.polar_loss(alpha, r_mats, T, G);

                        L_grads[op_name] = Lalpha_grad;
                        L_ave[i] += Lalpha_out;
                        
                    else:
                        
                        O = labels[op_name]
                        O_mat = self.op_matrices[i_m][op_name];
                        LO, LO_out = loss_calculator.O_loss(O, O_mat);
                        L_grads[op_name] = LO;
                        L_ave[i] += LO_out;
                        
                L = sum([op_names[key]*L_grads[key] \
                         for key in op_names])/self.n_molecules;
        
                L.backward()  # calculate the gradient

            self.optim.step()  # implement gradient descend

        if(self.device==0):
            torch.save(self.model.state_dict(), self.filename);

        return L_ave/steps/self.n_molecules;

