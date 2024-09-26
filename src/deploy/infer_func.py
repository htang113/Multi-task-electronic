import torch;
import json;
import numpy as np;
import os;
from deploy.apply import estimator;
from deploy.apply import load_data;
from basis import read_basis;
from train import recorder;
############## Setting parameters ################

def infer_func(params):

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

    loader = load_data(device=device, element_list = element_list,
                            path = data_path, 
                            starting_basis = 'def2-SVP');

    data, obs_mats = loader.load(datagroup);
        
    est = estimator(device, data, obs_mats);

    est.build_irreps(element_list = element_list);
    
    est.build_model(scaling = scaling);
    est.load(output_path + '/model/' + model_file);
    est.build_charge_matrices(data);

    rec = recorder(OPS, rank=rank, path = output_path);
    ############### Implement model training #####################

    properties = est.solve_apply(
                        batch_size = batch_size,
                        op_names=OPS);

    rec.save_apply(properties);


