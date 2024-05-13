#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:54:56 2023

@author: ubuntu
"""

from pkgs.integral import integrate
from pkgs.model import V_theta
from pkgs.predictor import predict_fns
import numpy as np
import torch
from pkgs.tomat import to_mat;
import os;
import json;
import matplotlib;
import matplotlib.pyplot as plt;

class estimator():

    def __init__(self, device, output_folder='test_output', scaling = 0.1) -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.integrator = integrate(device)
        self.loss = torch.nn.MSELoss();
        self.transformer = to_mat(device);
        self.output_folder = output_folder;
        self.scaling = scaling;

        if(not os.path.exists(output_folder)):
            os.mkdir(output_folder);
    
    def load(self, filename):

        self.model = V_theta(self.device).to(self.device)
        
        try:
            if(self.device=='cpu'):
                self.model.load_state_dict(torch.load(filename, map_location=torch.device(self.device)));
            else:
                self.model.load_state_dict(torch.load(filename));
        except:
            if(self.device=='cpu'):
                res = torch.load(filename, map_location=torch.device(self.device));
            else:
                res = torch.load(filename);
            for key in list(res.keys()):
                res[key[7:]] = res[key];
                del res[key];
            self.model.load_state_dict(res);

    def set_op_matrices(self, op_matrices):
        self.op_matrices = op_matrices

    def solve(self, minibatch, labels, obs_mat, E_nn, data_in,
              op_names=None, Cmat = [], save_filename='data') -> float:

        angstron2Bohr = 1.88973
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
        G *= self.scaling['G'];

        pred = predict_fns(h, V, ne, nbasis, nframe, self.device);

        elements = data_in['elements'];
        nuclearCharge = [1+5*(ele=='C') for ele in elements];
        nuclearCharge = torch.tensor(nuclearCharge, dtype=torch.float).to(self.device);
        mass = torch.tensor([1.008 + (12.011-1.008)*(ele=='C') for ele in elements], dtype=torch.float).to(self.device);
        pos = data_in['pos'];
        mass_center = torch.sum(pos*mass[None,:,None], axis=1)/torch.sum(mass);
        pos = (pos - mass_center[:,None,:])*angstron2Bohr;

        obj = {};

        for i, op_name in enumerate(op_names):

            if(op_name == 'E'):

                Ehat = pred.E(E_nn);
                obj[op_name] = {'E':(E+E_nn).tolist(), 'Ehat':Ehat.tolist()};

            if(op_name in ['x','y','z','xx','yy','zz','xy','xz','yz']):

                moment = torch.tensor([op_name.count('x'),
                            op_name.count('y'),
                            op_name.count('z')], dtype=torch.float).to(self.device);
                multipole = torch.sum(torch.prod(pos**moment[None,None,:],
                                                 axis=2)*nuclearCharge[None,:], axis=1);
                
                O = labels[op_name]
                O_mat = obs_mat[op_name]

                Ohat = pred.O(O_mat)
                O = multipole - O;
                Ohat = multipole - Ohat;
                obj[op_name] = {'Ohat': Ohat.tolist(),
                                'O': O.tolist()};
        
            if(op_name == 'atomic_charge'):

                C = labels['atomic_charge'];
                Chat = pred.C(Cmat);

                Chat = nuclearCharge[None,:]-Chat;
                C  = nuclearCharge[None,:]-C;
                obj['atomic_charge'] = {'Chat': Chat.tolist(),
                                        'C': C.tolist()};

            if(op_name == 'E_gap'):

                Eg = labels['E_gap'];
                Eg_hat = pred.Eg(G);
                obj['E_gap'] = {'Eg': Eg.tolist(),
                                'Eg_hat': Eg_hat.tolist()};

            if(op_name == 'bond_order'):
                    
                B = labels['B'];
                Bhat = pred.B(Cmat);
                obj['bond_order'] = {'B': B.tolist(),
                                    'Bhat': Bhat.tolist()};
            
            if(op_name == 'alpha'):

                alpha = labels['alpha'];
                r_mats = torch.stack([obs_mat['x'],
                                      obs_mat['y'],
                                      obs_mat['z']]);
                alpha_hat = pred.alpha(r_mats, T, G);
                obj['alpha'] = {'alpha': alpha.tolist(),
                                'alpha_hat': alpha_hat.tolist()};

        with open(self.output_folder+'/'+save_filename+'.json','w') as file:
            json.dump(obj,file)

        return Ehat, E;

    def plot(self, molecule_list, nrows=1):

        font = {'size' : 18}
        
        ncols = int((len(molecule_list)-1)//nrows)+1; 
        
        matplotlib.rc('font', **font)
        plt.figure(figsize=(6*ncols,5.5*nrows));
        res = [];
        res1 = [];
        for i in range(len(molecule_list)):

            with open(self.output_folder+'/'+molecule_list[i]+'.json','r') as file:

                data = json.load(file);

            E = data['E']['E'];
            Ehat = data['E']['Ehat'];

            plt.subplot(nrows,ncols,i+1);
            
            plt.plot([np.min(E)*27.211,np.max(E)*27.211], [np.min(E)*27.211,np.max(E)*27.211], 
                     linestyle='dashed', linewidth = 3,c='red');
            plt.scatter(np.array(E)*27.211,np.array(Ehat)*27.211, alpha=0.7);
            
            plt.title(molecule_list[i]);
            plt.xlabel('E$_{CCSDT}$  (eV)');
            plt.ylabel('E$_{NN}$  (eV)');
            res.append(np.mean((np.array(E)-np.array(Ehat))**2));

            if(molecule_list[i]=='C10H10'):
                natm = 20;
            else:
                if(molecule_list[i]=='CH4'):
                    atm_str = 'C1H4';
                else:
                    atm_str = molecule_list[i];
                natm =(int(atm_str[1])+int(atm_str[3:]));

            res1.append(np.mean((np.array(E)-np.array(Ehat))**2)/natm**2);
        plt.tight_layout();

        print('Standard deviation error:');
        print(str(np.sqrt(np.mean(res))/1.594*10**3)+' kcal/mol');
        print(str(np.sqrt(np.mean(res1))/1.594*10**3)+' kcal/mol/atom');
        plt.savefig(self.output_folder+'/E.png');
        plt.close();
        ops_dic = {'dipole':['x','y','z'],
                   'quadrupole':['xx','yy','zz','xy','xz','yz']};
        
        for ops in ops_dic:
            font = {'size' : 28}
            
            ncols = int((len(molecule_list)-1)//nrows)+1; 
            
            matplotlib.rc('font', **font)
            plt.figure(figsize=(15,10));
            res = [];
            
            if(ops=='dipole'):
                unit = 'Debye';
                plt.plot([-10,10], [-10,10], 
                         linestyle='dashed', linewidth = 3,c='red');
                plt.axis([-10,10,-10,10]);
                plt.title('Dipole moment');
            else:
                unit = 'e*a$_0^2$';
                plt.plot([-1000,1000], [-1000,1000], 
                         linestyle='dashed', linewidth = 3,c='red');
                plt.title('Quadrupole moment');
                plt.axis([-1000,1000,-1000,1000]);

            for i in range(len(molecule_list)):

                with open(self.output_folder+'/'+molecule_list[i]+'.json','r') as file:

                    data = json.load(file);

                O, Ohat = [], [];
                for opsi in ops_dic[ops]:
                    O += data[opsi]['O'];
                    Ohat += data[opsi]['Ohat'];
            
                plt.scatter(np.array(O),np.array(Ohat), alpha=0.7, c='blue');
                res.append(np.mean((np.array(O)-np.array(Ohat))**2));

            plt.xlabel(ops+'$_{CCSDT}$  '+unit);
            plt.ylabel(ops+'$_{NN}$  '+unit);

            print(ops+': Standard deviation error:');
            print(str(np.sqrt(np.mean(res)))+' '+unit);
            plt.savefig(self.output_folder+'/'+ops+'.png');
            plt.close();
        
        if('atomic_charge' in data):
            font = {'size' : 28}
            
            ncols = int((len(molecule_list)-1)//nrows)+1; 
            
            matplotlib.rc('font', **font)
            plt.figure(figsize=(15,10));
            res = [];
            plt.plot([-0.6, 0.6], [-0.6, 0.6], 
                        linestyle='dashed', linewidth = 3,c='red');

            for i in range(len(molecule_list)):

                with open(self.output_folder+'/'+molecule_list[i]+'.json','r') as file:

                    data = json.load(file);

                C, Chat = [], [];
                for u in range(len(data['atomic_charge']['C'])):
                    C += data['atomic_charge']['C'][u];
                    Chat += data['atomic_charge']['Chat'][u];

                plt.scatter(np.array(C),np.array(Chat),alpha=0.2,c='blue');
                res.append(np.mean((np.array(C)-np.array(Chat))**2));

            plt.title('atomic charge');
            plt.xlabel('atomic charge$_{CCSDT}$ (e)');
            plt.ylabel('atomic charge$_{NN}$ (e)');
            
            plt.axis([-0.6,0.6,-0.6,0.6])

            print('atomic charge: Standard deviation error:');
            print(str(np.sqrt(np.mean(res)))+' e');
            plt.savefig(self.output_folder+'/atomic_charge.png');
            plt.close();

        if('bond_order' in data):
            font = {'size' : 28}
            
            ncols = int((len(molecule_list)-1)//nrows)+1; 
            
            matplotlib.rc('font', **font)
            plt.figure(figsize=(15,10));
            res = [];
            plt.plot([0,3], [0,3], 
                         linestyle='dashed', linewidth = 3,c='red');
            for i in range(len(molecule_list)):

                with open(self.output_folder+'/'+molecule_list[i]+'.json','r') as file:

                    data = json.load(file);

                B, Bhat = [], [];
                for u in range(len(data['bond_order']['B'])):
                    for q1 in range(len(data['bond_order']['B'][u])):
                        for q2 in range(q1):
                            B_ele = data['bond_order']['B'][u][q1][q2];

                            B.append(B_ele);
                            Bhat.append(data['bond_order']['Bhat'][u][q1][q2]);

                plt.scatter(np.array(B),np.array(Bhat),alpha=0.2,c='blue');
                res.append(np.mean((np.array(B)-np.array(Bhat))**2));

            plt.title('bond order');
            plt.xlabel('B$_{CCSDT}$');
            plt.ylabel('B$_{NN}$');
            plt.axis([0,3,0,3])

            print('bond order: Standard deviation error:');
            print(str(np.sqrt(np.mean(res))));
            plt.savefig(self.output_folder+'/bond_order.png');
            plt.close();

        if('alpha' in data):
            font = {'size' : 28}
            
            ncols = int((len(molecule_list)-1)//nrows)+1; 
            
            matplotlib.rc('font', **font)
            plt.figure(figsize=(15,10));
            res = [];
            
            for i in range(len(molecule_list)):

                with open(self.output_folder+'/'+molecule_list[i]+'.json','r') as file:

                    data = json.load(file);

                alpha, alpha_hat = np.array(data['alpha']['alpha']).reshape(-1), np.array(data['alpha']['alpha_hat']).reshape(-1);
                res.append(np.mean((np.array(alpha)-np.array(alpha_hat))**2));
                plt.scatter(np.array(alpha),np.array(alpha_hat), alpha=0.2,c='blue');
            
            plt.plot([-55,150], [-55,150], 
                        linestyle='dashed', linewidth = 3,c='red');
            plt.axis([-55,150,-55,150]);
            plt.title('polarizability');
            plt.xlabel('$alpha_{CCSDT}$ (a.u.)');
            plt.ylabel('$alpha_{NN}$ (a.u.)');

            plt.tight_layout();

            print('polarizability: Standard deviation error:');
            print(str(np.sqrt(np.mean(res)))+ ' (a.u.)');
            plt.savefig(self.output_folder+'/polarizability.png');
            plt.close();

        if('E_gap' in data):
            font = {'size' : 28}
            
            ncols = int((len(molecule_list)-1)//nrows)+1; 
            
            matplotlib.rc('font', **font)
            plt.figure(figsize=(15,10));
            res = [];

            plt.plot([0,12.5], [0,12.5], 
                        linestyle='dashed', linewidth = 3,c='red');            
            for i in range(len(molecule_list)):

                with open(self.output_folder+'/'+molecule_list[i]+'.json','r') as file:

                    data = json.load(file);
                hartree_to_eV = 27.211386245988;
                Eg, Eghat = np.array(data['E_gap']['Eg'])*hartree_to_eV, np.array(data['E_gap']['Eg_hat'])*hartree_to_eV;

                plt.scatter(np.array(Eg),np.array(Eghat), alpha=0.2,c='blue');
                res.append(np.mean((np.array(Eg)-np.array(Eghat))**2));

            plt.axis([0,12.5,0,12.5])
            plt.title('Energy gap');
            plt.xlabel('Eg$_{CCSDT}$ (eV)');
            plt.ylabel('Eg$_{NN}$ (eV)');

            print('Energy gap: Standard deviation error:');
            print(str(np.sqrt(np.mean(res)))+ ' eV');
            plt.savefig(self.output_folder+'/energy_gap.png');
            plt.close();