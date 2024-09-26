# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:00:28 2023

@author: 17000
"""

from deploy.apply import load_data;
from deploy.apply import estimator;
from deploy.apply import sampler;
import torch;
import json;
import time;

device = 'cpu';
scaling = {'V':0.2, 'T': 0.01};

path = '/path/to/data/';
starting_basis = 'cc-pVDZ';

OPS = {'V':0.1,'E':1,
       'x':0.1, 'y':0.1, 'z':0.1,
       'xx':0.1, 'yy':0.1, 'zz':0.1,
       'xy':0.1, 'yz':0.1, 'xz':0.1,
       'atomic_charge': 0.1, 'E_gap':0.1,
       'bond_order':0.1, 'alpha':0.0};

operators_electric = [key for key in list(OPS.keys()) \
                      if key in ['x','y','z','xx','yy',
                                 'zz','xy','xz','yz']];

with open(path + 'B2_data.json','r') as file:
    data1 = json.load(file);
molecule_list = data1['name'];

est = estimator(device, starting_basis = starting_basis, scaling = scaling);
est.load('EGNN_hydrocarbon_model.pt');

data, labels, obs_mat = load_data(data1, device, starting_basis=starting_basis, op_names=operators_electric);

tl = [];
sampler1 = sampler(data, labels, device);

ind_list = list(range(len(molecule_list)));
for i in ind_list:
    
    print('Solving '+str(i)+'th molecule: '+molecule_list[i])
    start = time.time();

    E_nn = labels[i]['E_nn'];

    elements = data[i]['elements'];
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
    Cmat = torch.hstack(mati);
    
    minibatch, labels1 = sampler1.sample(batch_size=1, i_molecule=i,
                                        op_names=operators_electric);

    Ehat = est.solve(minibatch, labels1, obs_mat[i], E_nn, data[i],
                        save_filename=molecule_list[i],
                        op_names = list(OPS.keys()), Cmat = Cmat);

    tl.append(time.time() - start);

#    with open('test_output/'+molecule_list[i]+'.json','r') as file:
#        data2 = json.load(file);
#    data2['correction'] = data1['correction'][i];
#    with open('test_output/'+molecule_list[i]+'.json','w') as file:
#        json.dump(data2,file);

with open('time.json','w') as file:
    json.dump(tl[3:],file);