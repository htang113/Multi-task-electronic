import torch;
import json;
import numpy as np;
import os;
from deploy.apply import estimator;
from data.interface_pyscf import generate_basic
from basis import read_basis;
from train import recorder;
############## Setting parameters ################

def pyscf_func(params, elements, pos, name):

    world_size = 1;
    rank = 0;
    batch_size = 1;

    OPS = {'E':1,'x':0.1, 'y':0.1, 'z':0.1,
           'xx':0.01, 'yy':0.01, 'zz':0.01,
           'xy':0.01, 'yz':0.01, 'xz':0.01,
           'atomic_charge': 0.01, 'E_gap':0.2,
           'bond_order':0.02, 'alpha':3E-5};

    device = params['device'];
    scaling = params['scaling'];
    element_list = params['element_list'];
    path = params['path'];
    model_file = params['model_file'];
    output_path = params['output_path'];

    ###################### load data and model ######################

    basis_path = path + 'script';
    read_basis(path = basis_path, savefile=True);

    dftcalc = generate_basic(device, path, element_list);
    data, obs_mats = dftcalc.generate(elements, pos, name);
        
    est = estimator(device, [data], [obs_mats]);

    est.build_irreps(element_list = element_list);
    
    est.build_model(scaling = scaling);
    est.load(output_path + '/model/' + model_file);
    est.build_charge_matrices([data]);

    rec = recorder(OPS, rank=rank, path = output_path);
    ############### Implement model training #####################

    properties = est.solve_apply(
                        batch_size = batch_size,
                        op_names=OPS);
    rec.inference_file = rec.inference_file[:-14] + name + '.json';
    rec.save_apply(properties);

    return properties;


