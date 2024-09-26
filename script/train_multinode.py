import torch;
import json;
import numpy as np;
from train import trainer_ddp;
from data import load_data;
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
import os;

def training():

    # create default process group
    dist.init_process_group('gloo');
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()

    # create local model
    
    OPS = {'V':0.1,'E':1,
       'x':0.2, 'y':0.2, 'z':0.2,
       'xx':0.01, 'yy':0.01, 'zz':0.01,
       'xy':0.01, 'yz':0.01, 'xz':0.01,
       'atomic_charge': 0.01, 'E_gap':0.2,
       'bond_order':0.02, 'alpha':3E-5};
    
    batch_size = 496;
    steps_per_epoch = 10;
    N_epoch = 401;
    lr_init = 1E-2;
    lr_final = 1E-3;
    lr_decay_steps = 50;
    scaling = {'V':0.2, 'T': 0.01};
    Nsave = 40;

    path = '/pscratch/sd/t/th1543/v6/data/';

    molecule_list = [['CH4'] , ['C3H8'], ['C4H8'],  ['C7H8'],  ['C6H12'],
                     ['C2H2'], ['C4H6'], ['C4H10'], ['C5H12'], ['C7H10'],
                     ['C2H4'], ['C3H6'], ['C5H8'],  ['C6H8'],  ['C8H8'],
                     ['C2H6'], ['C3H4'], ['C6H6'],  ['C5H10'], ['C6H14']];

    operators_electric = [key for key in list(OPS.keys()) \
                          if key in ['x','y','z','xx','yy',
                                     'zz','xy','xz','yz']];
        
    data, labels, obs_mats = load_data(molecule_list[rank], device, 
                                       path=path,
                                       op_names=operators_electric,
                                       ind_list = [i for i in range(batch_size) if i%4<=2]);

    train1 = trainer_ddp(device, data, labels, lr=lr_init,
                         filename=str(rank)+'_model.pt',
                         op_matrices=obs_mats, scaling=scaling);

    if(os.path.exists('0_model.pt')):
        train1.load('0_model.pt');

    scheduler = StepLR(train1.optim, step_size=lr_decay_steps,
                       gamma=(lr_final/lr_init)**(lr_decay_steps/N_epoch));
    
    # define loss function and optimizer
    with open(str(rank)+'.txt','w') as file:
        file.write('epoch\t');
        for i in range(len(OPS)):
            file.write(' loss_'+str(list(OPS.keys())[i])+'\t');
        file.write('\n');
        
    # forward pass
    for i in range(N_epoch):
        
        loss = train1.train(steps=steps_per_epoch,
                            batch_size = batch_size,
                            op_names=OPS);
        
        scheduler.step();

        with open(str(rank)+'.txt','a') as file:
            file.write(str(i)+'\t')
            for j in range(len(loss)):
                file.write(str(loss[j])+'\t');
            file.write('\n');

        if(i%Nsave == 0 and i>0 and rank==0):
            train1.save(str(i)+'_model.pt');
            print('saved model at epoch '+str(i));
    dist.destroy_process_group()

training()
    

