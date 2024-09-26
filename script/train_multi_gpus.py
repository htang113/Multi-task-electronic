import torch;
import json;
import numpy as np;
from train import trainer_ddp;
from data import data_loader;
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
import os;

def training(rank, world_size):

    # create default process group
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '29500')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    
    OPS = {'V':0.1,'E':1,
       'x':0.2, 'y':0.2, 'z':0.2,
       'xx':0.01, 'yy':0.01, 'zz':0.01,
       'xy':0.01, 'yz':0.01, 'xz':0.01,
       'atomic_charge': 0.01, 'E_gap':0.2,
       'bond_order':0.02, 'alpha':3E-5};
    
    batch_size = 100;
    steps_per_epoch = 1;
    N_epoch = 101;
    lr_init = 1E-3;
    lr_final = 1E-3;
    lr_decay_steps = 50;
    scaling = {'V':0.2, 'T': 0.01};
    Nsave = 50;
    element_list = ['H','C','N','O','F'];

    path = '/pscratch/sd/t/th1543/v2.0/data';

    loader = data_loader(device=rank, element_list = element_list,
                            path = path, batch_size = batch_size, starting_basis = 'def2-SVP');

    data, labels, obs_mats = loader.load_data('group1');

    train1 = trainer_ddp(rank, data, labels, lr=lr_init,
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

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    world_size = 4;
    mp.spawn(training,
        args=(world_size,),
        nprocs=world_size,
        join=True)
    

