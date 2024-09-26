# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:00:28 2023

@author: 17000
"""

from data.dataframe import load_data;
from deploy.deploy import estimator;
from model.sample_minibatch import sampler;
import torch;

device = 'cuda:0';
scaling = {'V':0.2, 'T': 0.01, 'G':1};
batch_size = [496]*20;
batch_size += [250, 150, 98];

molecule_list = ['CH4','C2H2','C2H4','C2H6','C3H4',
                 'C3H6','C3H8','C4H6','C4H8','C4H10',
                 'C5H8','C5H10','C5H12','C6H6','C6H8',
                 'C6H12','C6H14','C7H8','C7H10','C8H8'];

molecule_list += ['C7H14', 'C8H14', 'C10H10'];

path = '/path/to/data';

OPS = {'V':0.1,'E':1,
       'x':0.1, 'y':0.1, 'z':0.1,
       'xx':0.1, 'yy':0.1, 'zz':0.1,
       'xy':0.1, 'yz':0.1, 'xz':0.1,
       'atomic_charge': 0.1, 'E_gap':0.1,
       'bond_order':0.1, 'alpha':0.0};

operators_electric = [key for key in list(OPS.keys()) \
                      if key in ['x','y','z','xx','yy',
                                 'zz','xy','xz','yz']];

est = estimator(device, scaling = scaling);
est.load('EGNN_hydrocarbon_model.pt');

for i in range(len(molecule_list)):
    
    print('Solving '+str(i)+'th molecule: '+molecule_list[i])

    if(batch_size[i]==496):
        ind = [j for j in range(496) if j%4==3];
    else:
        ind = range(batch_size[i]);

    data, labels, obs_mat = load_data(molecule_list[i:i+1], device, path=path,
                            ind_list=ind,op_names=operators_electric
                            );

    sampler1 = sampler(data, labels, device);

    E_nn = labels[0]['E_nn'];

    elements = data[0]['elements'];
    orbitals_list = [9*(u=='C')+5 for u in elements];
    map1 = [sum(orbitals_list[:j]) for j in range(len(elements)+1)];
    mati = [];
    for j in range(len(elements)):
        Sm = torch.zeros_like(labels[0]['Smhalf']);
        Sm[:,map1[j]:map1[j+1],:] = \
        labels[0]['Smhalf'][:,map1[j]:map1[j+1],:];
        S = torch.matmul(torch.matmul(labels[0]['Smhalf'], Sm),
                     labels[0]['S']);
        mati.append(S[:,None,:,:]);
    Cmat = torch.hstack(mati);
    
    minibatch, labels1 = sampler1.sample(batch_size=batch_size[i], i_molecule=0,
                                        op_names=operators_electric);

    Ehat, E = est.solve(minibatch, labels1, obs_mat[0], E_nn, data[0],
                        save_filename=molecule_list[i],
                        op_names = list(OPS.keys()), Cmat = Cmat);

#est.plot(molecule_list, nrows=4)

