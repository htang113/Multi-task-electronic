#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:39:10 2023

@author: ubuntu
"""

import json;
import torch;
import scipy;

def load_data(molecules, device, path = 'data', load_obs_mat = True, 
              ind_list = [], op_names = []):

    data_in = [];
    labels = [];
    obs_mats = [];
    
    cm_inverse_to_hartree = 4.55633528*1E-6;
    
    for molecule in molecules:
        with open(path+'/'+molecule+'_data.json', 'r') as file:

            data = json.load(file);

        if(ind_list == []):

            pos = torch.tensor(data['coordinates']).to(device);
            pos[:,:,[0,1,2]] = pos[:,:,[1,2,0]];
            elements = data['elements'][0];
            nuclearCharge = [1+5*(ele=='C') for ele in elements];
            nuclearCharge = torch.tensor(nuclearCharge, dtype=torch.float).to(device);
            
            ne = int(round(sum([1+5*(ele=='C') for ele in elements])/2));
            norbs = int(round(sum([5+9*(ele=='C') for ele in elements])));
            nframe = len(pos);

            data_in.append({'pos':pos,'elements':elements,
                       'properties':{'ne':ne, 'norbs':norbs,'nframe':nframe}});

            h = torch.tensor(data['h']).to(device);
            S_mhalf = [scipy.linalg.fractional_matrix_power(si, (-1/2)).tolist() for si in data['S']]
            S_mhalf = torch.tensor(S_mhalf).to(device);
            h = torch.matmul(torch.matmul(S_mhalf, h),S_mhalf);
            
            B_labels = torch.zeros([nframe, len(elements), 
                                    len(elements)]).to(device);
            for u in range(nframe):
                for dp in data['bond_order'][u]:
                    B_labels[u, dp[0], dp[1]] = dp[2];
                    B_labels[u, dp[1], dp[0]] = dp[2];
            
            label = {'S': torch.tensor(data['S']).to(device),
                     'Smhalf': S_mhalf,
                     'h': h,
                     'E': torch.tensor(data['energy']).to(device),
                     'E_nn':torch.tensor(data['Enn']).to(device),
                     'atomic_charge': - torch.tensor(data['atomic_charge']).to(device) + \
                          nuclearCharge[None,:],
                     'E_gap': cm_inverse_to_hartree * \
                         torch.tensor([u[0] for u in data['Ee']]).to(device),
                     'B': B_labels,
                     'alpha': torch.tensor(data['alpha']).to(device)};
            
            for op_name in op_names:
                label[op_name] = torch.tensor(data[op_name]).to(device)
            labels.append(label);
            
            if(load_obs_mat):
                with open(path + '/'+molecule+'_obs_mat.json', 'r') as file:
                    data_obs = json.load(file);
    
                if op_names is None:
                    op_names = set(data_obs.keys())
    
                obs_mats.append({op: torch.matmul(torch.matmul(S_mhalf,torch.Tensor(data_obs[op]).to(device)),S_mhalf)
                                                                for op in op_names})
                
        else:
            
            pos = torch.tensor(data['coordinates'])[ind_list].to(device);
            pos[:,:,[0,1,2]] = pos[:,:,[1,2,0]];
            elements = data['elements'][0];
            nuclearCharge = [1+5*(ele=='C') for ele in elements];
            nuclearCharge = torch.tensor(nuclearCharge, dtype=torch.float).to(device);
            
            ne = int(round(sum([1+5*(ele=='C') for ele in elements])/2));
            norbs = int(round(sum([5+9*(ele=='C') for ele in elements])));
            nframe = len(pos);

            data_in.append({'pos':pos,'elements':elements,
                       'properties':{'ne':ne, 'norbs':norbs,'nframe':nframe}});

            h = torch.tensor(data['h'])[ind_list].to(device);
            S_mhalf = [scipy.linalg.fractional_matrix_power(data['S'][si], (-1/2)).tolist() for si in ind_list]
            S_mhalf = torch.tensor(S_mhalf).to(device);
            h = torch.matmul(torch.matmul(S_mhalf, h),S_mhalf);
            
            B_labels = torch.zeros([nframe, len(elements), 
                                    len(elements)]).to(device);
            for u in range(nframe):
                for dp in data['bond_order'][ind_list[u]]:
                    B_labels[u, dp[0], dp[1]] = dp[2];
                    B_labels[u, dp[1], dp[0]] = dp[2];
                    
            label = {'S': torch.tensor(data['S'])[ind_list].to(device),
                     'Smhalf': S_mhalf,
                     'h': h,
                     'E': torch.tensor(data['energy'])[ind_list].to(device),
                     'E_nn':torch.tensor(data['Enn'])[ind_list].to(device),
                     'atomic_charge': -torch.tensor(data['atomic_charge'])[ind_list].to(device) + \
                          nuclearCharge[None,:],
                     'E_gap': cm_inverse_to_hartree * \
                          torch.tensor([u[0] for u in data['Ee']])[ind_list].to(device),
                     'B': B_labels,
                     'alpha': torch.tensor(data['alpha'])[ind_list].to(device)};
                
            for op_name in op_names:
                label[op_name] = torch.tensor(data[op_name])[ind_list].to(device)
            labels.append(label);
            
            if(load_obs_mat):
                with open(path+'/'+molecule+'_obs_mat.json', 'r') as file:
                    data_obs = json.load(file);                  
    
                if op_names is None:
                    op_names = set(data_obs.keys())
    
                obs_mats.append({op: torch.matmul(torch.matmul(S_mhalf,torch.Tensor(data_obs[op])[ind_list].to(device)),S_mhalf)
                                                                for op in op_names})
                
    return data_in, labels, obs_mats;


