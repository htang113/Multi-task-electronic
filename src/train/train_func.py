import torch;
import json;
import numpy as np;
from torch.optim.lr_scheduler import StepLR;
import os;

import torch.distributed as dist;
from torch.nn.parallel import DistributedDataParallel as DDP;
import torch.multiprocessing as mp

from train import trainer, recorder;
from data import dataloader;
from model import V_theta;
from basis import read_basis;

############## Setting parameters ################

def train_func(rank, params):

    OPS = params['OPS'];
    device = params['device'];
    batch_size = params['batch_size'];
    steps_per_epoch = params['steps_per_epoch'];
    N_epoch = params['N_epoch'];
    lr_init = params['lr_init'];
    lr_final = params['lr_final'];
    lr_decay_steps = params['lr_decay_steps'];
    scaling = params['scaling'];
    Nsave = params['Nsave'];

    element_list = params['element_list'];
    path = params['path'];
    output_path = params['output_path'];
    datagroup = params['datagroup'];
    load_model = params['load_model'];

    world_size = params['world_size'];
    
    ###################### load data and model ######################

    basis_path = path + 'script';
    data_path = path + 'dataset';
    read_basis(path = basis_path, savefile=True);

    loader = dataloader(device=device, element_list = element_list,
                            path = data_path, batch_size = batch_size, 
                            starting_basis = 'def2-SVP');

    data, labels, obs_mats, batch_size = loader.load_data(datagroup, rank, world_size);
        
    train1 = trainer(device, data, labels,
                    op_matrices=obs_mats);

    train1.build_irreps(element_list = element_list);
    
    if(world_size>1):
        train1.build_ddp_model(scaling = scaling);
    else:
        train1.build_model(scaling = scaling);
    
    train1.build_optimizer(lr=lr_init);
    train1.build_charge_matrices(data);

    if(load_model):
        train1.load('model.pt');

    scheduler = StepLR(train1.optim, step_size=lr_decay_steps,
                        gamma=(lr_final/lr_init)**(lr_decay_steps/N_epoch));

    rec = recorder(OPS, rank=rank, path = output_path);
    ############### Implement model training #####################
    for i in range(N_epoch):
        
        loss = train1.train(steps=steps_per_epoch,
                            batch_size = batch_size,
                            op_names=OPS);
        
        scheduler.step();
        rec.record_loss(i, loss);
        rec.save_model(train1, i, Nsave);

def train_spawn(rank, params):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=params['world_size'])
    train_func(rank, params);
    dist.destroy_process_group()

    return None;

def train_serial(params):

    train_func(0, params);

    return None;

def train_elastic(params):

    dist.init_process_group('gloo');
    rank = dist.get_rank();
    device = rank % torch.cuda.device_count();
    params['device'] = device;

    train_func(rank, params);

    dist.destroy_process_group()

    return None;

def main(params):

    if(params['world_size'] == 1):
        train_serial(params);

    elif(params['ddp_mode'] == 'spawn'):
        mp.spawn(train_spawn,
            args=(params,),
            nprocs=params['world_size'],
            join=True);

    elif(params['ddp_mode'] == 'elastic'):

        train_elastic(params);
    
    else:
        raise ValueError('Invalid ddp_mode');
    
    return None;