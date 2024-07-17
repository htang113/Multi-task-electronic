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

We include 3 demo scripts in demo/ for training the EGNN model and using our pre-trained model to calculate molecular properties. 

3.1 Demo for training a model

The training demo/demo_train.py script is shown below:
```
from mtelect import train_model

# Define the operators and their weights in the training loss
OPS = {'V':0.1,'E':1,
    'x':0.2, 'y':0.2, 'z':0.2,
    'xx':0.01, 'yy':0.01, 'zz':0.01,
    'xy':0.01, 'yz':0.01, 'xz':0.01,
    'atomic_charge': 0.01, 'E_gap':0.2,
    'bond_order':0.04, 'alpha':3E-5};
# Specify the chemical formula for training. Each chemical formula corresponds to a separate file in training dataset.
molecule_list = ['CH4' ,'C3H8', 'C4H8', 'C7H8', 'C6H12',
                 'C2H2', 'C4H6','C4H10', 'C5H12', 'C7H10',
                 'C2H4', 'C3H6','C5H8', 'C6H8', 'C8H8',
                 'C2H6','C3H4', 'C6H6', 'C5H10', 'C6H14'];
device = 'cuda:0';     # device to train the model on. Set to 'cpu' to train on CPU
steps_per_epoch = 10;  # number of training steps per epoch
N_epoch = 21;         # number of epochs to train for
lr_init = 1E-2;        # initial learning rate
lr_final = 1E-3;       # final learning rate
lr_decay_steps = 5;    # number of epochs to decay the learning rate  
scaling = {'V':0.2, 'T': 0.01};   # scaling factors for the neural network output
                                           # for V_theta, screening matrix T, and bandgap corrector G
Nsave = 50;      # number of epochs between saving the model
batch_size = 1;  # number of configurations from each chemical formula for training
path = 'data/';  # path to training data

world_size = 1;  # number of GPUs to train on. Set to 1 to train on a single GPU or CPU
rank = 0;        # rank of the current GPU. Set to 0 for single GPU or CPU training

params = {'OPS':OPS, 'molecule_list':molecule_list, 'device':device,
            'steps_per_epoch':steps_per_epoch, 'N_epoch':N_epoch,
            'lr_init':lr_init, 'lr_final':lr_final, 'lr_decay_steps':lr_decay_steps,
            'scaling':scaling, 'Nsave':Nsave, 'batch_size':batch_size, 'path':path,
            'world_size':world_size, 'rank':rank};

train_model(params);
```
A small training dataset is in the "data" folder, including starting-point DFT Hamiltonian and CCSD(T) labels of 23 molecules at equilibrium configuration. The above script can be launched by the following commands:
```
cd demo
cp demo_train.py ../
cd ../
python3 demo_train.py
```
In the repository folder. The training takes 10~20 minutes on a normal Desktop computer. Running the program writes the loss data into the loss_0.txt file and output the trained model as "model.pt". Two additional files named "10_model.pt" and "20_model.pt" are saved as checkpoints at the 10 and 20 epoch, respectively. The loss data looks as below, including the mean square error loss of all trained quantities (V: the $\parallel V_\theta\parallel^2$ regularization, E: energy, x/y/z:different components of electric dipole moments, xx/yy/zz/xy/yz/xz: different components of electric quadrupole moments, atomic_charge: Mulliken atomic charge, bond_order: Mayer bond order, alpha: static electric polarizability)

![image](https://github.com/user-attachments/assets/5570cdf5-5e0e-4249-9e1e-78ec303cca98)

Depending on the random seed, specific numbers can be different, but the decreasing trend of the training loss is expected in all cases.

The training can also be implemented on multiple GPU's in parallel using the distributed data parallel (DDP) scheme. The DDP version of the training script is shown below:
```
from mtelect import train_model
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os;

def run(rank, world_size):

    # set environment and create default process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    OPS = {'V':0.1,'E':1,
       'x':0.2, 'y':0.2, 'z':0.2,
       'xx':0.01, 'yy':0.01, 'zz':0.01,
       'xy':0.01, 'yz':0.01, 'xz':0.01,
       'atomic_charge': 0.01, 'E_gap':0.2,
       'bond_order':0.02, 'alpha':3E-5};  # Define the operators and their weights in the training loss
    
    device = 'cuda:'+str(rank);  # device to train the model on for the current process
    batch_size = 1;              # number of configurations from each chemical formula for training
    steps_per_epoch = 10;        # number of training steps per epoch
    N_epoch = 21;                # number of epochs to train for
    lr_init = 1E-2;              # initial learning rate
    lr_final = 1E-3;             # final learning rate
    lr_decay_steps = 50;         # number of epochs to decay the learning rate
    scaling = {'V':0.2, 'T': 0.01};  # scaling factors for the neural network output
                                     # for V_theta and screening matrix T
    Nsave = 10;             # number of epochs between saving the model
    path = 'data/';         # path to training data
    # Specify the chemical formula for training.
    # Chemical formula are separated into 4 groups for parallel training on 4 GPUs.
    molecule_list = [['CH4' ,'C3H8', 'C4H8', 'C7H8', 'C6H12'],
                     ['C2H2', 'C4H6','C4H10', 'C5H12', 'C7H10'],
                     ['C2H4', 'C3H6','C5H8', 'C6H8', 'C8H8'],
                     ['C2H6','C3H4', 'C6H6', 'C5H10', 'C6H14']]; 

    params = {'OPS':OPS, 'molecule_list':molecule_list, 'device':device,
            'steps_per_epoch':steps_per_epoch, 'N_epoch':N_epoch,
            'lr_init':lr_init, 'lr_final':lr_final, 'lr_decay_steps':lr_decay_steps,
            'scaling':scaling, 'Nsave':Nsave, 'batch_size':batch_size, 'path':path,
            'world_size':world_size, 'rank':rank};
    
    # train the model
    train_model(params);

if __name__=="__main__":

    world_size = 4;  # number of GPUs to train on
    mp.spawn(run,
        args=(world_size,),
        nprocs=world_size,
        join=True)   # spawn 4 processes for parallel training on 4 GPUs
```
This script run the same training on 4 GPU's in parallel. If your device has a different number of GPUs, change "world_size" accordingly. Note that in the current version, molecules trained on each process must be manually separated by setting "molecule_list", and each process contains the same number of data files. Each process will output a loss file (loss_0,1,2,3.txt) including the MAE loss for data within the process.

3.2 Demo for using our pre-trained model 

The model inference script is shown below:

```
from mtelect import infer

device = 'cpu';  # device to run the inference calculation on
scaling = {'V':0.2, 'T': 0.01};  # scaling factors for the neural network output. 
                                 # should be set the same as in the training script
data_path = 'data/cyclic_PA_data.json';  # path to the data file of molecule to predict
model_path = 'models/EGNN_hydrocarbon_model.pt';  # path to the pre-trained model
output_path = 'output/';  # path to save the output files

OPS = ['E','x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'yz', 'xz',
       'atomic_charge', 'E_gap', 'bond_order', 'alpha'];      # list of operators to predict

params = {'device':device, 'scaling':scaling, 'data_path':data_path,
          'model_path':model_path, 'OPS':OPS, 'output_path':output_path};

infer(params);
```

4. Instructions for use
Our pre-trained model "EGNN_hydrocarbon_model.pt" is also included in the repository. In order to use the model to calculate new systems, 
