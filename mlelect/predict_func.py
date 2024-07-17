# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:00:28 2023

@author: 17000
"""

from mtelect.apply import load_data_apply as load_data;
from mtelect.apply import estimator_apply as estimator;
from mtelect.apply import sampler_apply as sampler;
import torch;
import json;

def infer(params):

    device = params['device'];
    scaling = params['scaling'];
    datapath = params['data_path'];
    model_path = params['model_path'];
    OPS = params['OPS'];
    output_path = params['output_path'];

    with open(datapath,'r') as file:
        data1 = json.load(file);
    molecule_name = data1['name'][0];

    est = estimator(device, scaling = scaling, output_folder = output_path);
    est.load(model_path);

    data, labels, obs_mat = load_data(data1, device);

    sampler1 = sampler(data, labels, device);
        
    print('Solving molecule: '+molecule_name)

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

    minibatch, labels1 = sampler1.sample(batch_size=1, i_molecule=0);

    Ehat = est.solve(minibatch, labels1, obs_mat[0], E_nn, data[0],
                        save_filename=molecule_name,
                        op_names = OPS, Cmat = Cmat);
