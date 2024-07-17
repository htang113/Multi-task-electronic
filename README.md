# Multi-task-electronic
This package provides a python realization of the multi-task EGNN (equivariant graph neural network) for molecular electronic structure described in the paper "Multi-task learning for molecular electronic structure approaching coupled-cluster accuracy".

1. System requirements

The package works in Linux and Windows systems. The following packages need to be installed in advance:

python==3.10.13
numpy=1.26.4
scipy==1.12.0
matplotlib==3.9.0
torch==2.1.0
torch_cluster==1.6.3
torch_scatter==2.1.2
e3nn==0.5.1
sympy==1.13.0

nvidia-dali-cuda120==1.35.0  (required for using GPU in the calculation)

Note that in most cases, different version of packages should also work. We list exactly the versions in our calculations in case version inconsistency issue occurs. If users intend to run the program on a cpu device, the cuda package is not needed.

2. Installation

No installation process is needed before running the calculation. After downloading the whole repository, calculations can be run in the folder. 
For example, one can launch the model training process using "python3 demo.py" in the terminal for Linux devices. 

3. Demo

We include a demo for training the EGNN model "demo.py". 
```
import torch;
import json;
import numpy as np;
from pkgs.train import trainer;
from pkgs.dataframe import load_data;
from pkgs.model import V_theta;
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
import os;

OPS = {'V':0.1,'E':1,
    'x':0.2, 'y':0.2, 'z':0.2,
    'xx':0.01, 'yy':0.01, 'zz':0.01,
    'xy':0.01, 'yz':0.01, 'xz':0.01,
    'atomic_charge': 0.01, 'E_gap':0.2,
    'bond_order':0.04, 'alpha':3E-5};

device = 'cuda:0';
steps_per_epoch = 10;
N_epoch = 21;
lr_init = 1E-2;
lr_final = 1E-3;
lr_decay_steps = 50;
scaling = {'V':0.2, 'T': 0.01, 'G':0.1};
Nsave = 10;
batch_size = 1;
path = '../../v6/data/demo/';

molecule_list = ['CH4' ,'C3H8', 'C4H8', 'C7H8', 'C6H12',
                 'C2H2', 'C4H6','C4H10', 'C5H12', 'C7H10',
                 'C2H4', 'C3H6','C5H8', 'C6H8', 'C8H8',
                 'C2H6','C3H4', 'C6H6', 'C5H10', 'C6H14'];


operators_electric = [key for key in list(OPS.keys()) \
                        if key in ['x','y','z','xx','yy',
                                    'zz','xy','xz','yz']];
data, labels, obs_mats = load_data(molecule_list, device, 
                                    path=path,
                                    op_names=operators_electric);

train1 = trainer(device, data, labels, lr=lr_init,
                        filename='model.pt',
                        op_matrices=obs_mats, scaling=scaling);

if(os.path.exists('model.pt')):
    train1.load('model.pt');

scheduler = StepLR(train1.optim, step_size=lr_decay_steps,
                    gamma=(lr_final/lr_init)**(lr_decay_steps/N_epoch));

# define loss function and optimizer
with open('loss.txt','w') as file:
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

    with open('loss.txt','a') as file:
        file.write(str(i)+'\t')
        for j in range(len(loss)):
            file.write(str(loss[j])+'\t');
        file.write('\n');

    if(i%Nsave == 0 and i>0):
        train1.save(str(i)+'_model.pt');
        print('saved model at epoch '+str(i));
```
A small training dataset is in the "data" folder, including starting-point DFT Hamiltonian and CCSD(T) labels of 23 molecules at equilibrium configuration. The above script can be launched by simply typing 
```
python3 demo.py
```
In the repository folder. The training takes about 10 minutes on a normal Desktop computer. Running the program gives 

4. Instructions for use

