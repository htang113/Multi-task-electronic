import torch;
import json;
import numpy as np;
import os;
from deploy import estimator_test;
from data import dataloader;
from basis import read_basis;
from train import recorder;
############## Setting parameters ################

def test_func(params):

    world_size = 1;
    rank = 0;
    OPS = params['OPS'];
    device = params['device'];
    batch_size = params['batch_size'];
    scaling = params['scaling'];

    element_list = params['element_list'];
    path = params['path'];
    output_path = params['output_path'];
    datagroup = params['datagroup'];
    model_file = params['model_file'];

    ###################### load data and model ######################

    basis_path = path + 'script';
    data_path = path + 'dataset';
    read_basis(path = basis_path, savefile=True);

    loader = dataloader(device=device, element_list = element_list,
                            path = data_path, batch_size = batch_size, 
                            starting_basis = 'def2-SVP');

    data, labels, obs_mats, batch_size = loader.load_data(datagroup, rank, world_size);
        
    estimator = estimator_test(device, data, labels,
                    op_matrices=obs_mats);

    estimator.build_irreps(element_list = element_list);
    
    estimator.build_model(scaling = scaling);
    estimator.load(output_path + '/model/' + model_file);
    estimator.build_charge_matrices(data);

    rec = recorder(OPS, rank=rank, path = output_path);
    ############### Implement model training #####################
        
    properties = estimator.solve(
                        batch_size = batch_size,
                        op_names=OPS);

    rec.save_test(properties);


