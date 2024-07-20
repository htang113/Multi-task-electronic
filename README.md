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

First, create a python virtual environment:
```
python3 -m venv ~/venv/MTElect
source ~/venv/MTElect/bin/activate
```
Then install dependent packages:
```
pip install numpy
pip install scipy
pip install matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install --upgrade e3nn
pip install sympy
```
After installing the dependent packages, download this code package:
```
git clone https://github.com/htang113/Multi-task-electronic
```
Finally enter the working folder and install the package:
```
cd Multi-task-electronic
pip install .
```

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
In the repository folder. The training takes 15~20 minutes on a normal Desktop computer. Running the program writes the loss data into the loss_0.txt file and output the trained model as "model.pt". Two additional files named "10_model.pt" and "20_model.pt" are saved as checkpoints at the 10 and 20 epoch, respectively. The loss data looks as below, including the mean square error loss of all trained quantities (V: the $\parallel V_\theta\parallel^2$ regularization, E: energy, x/y/z:different components of electric dipole moments, xx/yy/zz/xy/yz/xz: different components of electric quadrupole moments, atomic_charge: Mulliken atomic charge, bond_order: Mayer bond order, alpha: static electric polarizability)

![image](https://github.com/user-attachments/assets/5570cdf5-5e0e-4249-9e1e-78ec303cca98)

Depending on the random seed, specific numbers can be different, but the decreasing trend of the training loss is expected in all cases.

The training can also be implemented on multiple GPU's in parallel using the distributed data parallel (DDP) scheme in "demo/demo_parallel.py". The DDP version of the training script is shown below:
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
This script run the same training on 4 GPU's in parallel. This calculation takes about 5 min on the NERSC perlmutter GPU node with 4 Nvidia A100 40 GB GPUs https://docs.nersc.gov/systems/perlmutter/architecture/. If your device has a different number of GPUs, change "world_size" accordingly. Note that in the current version, molecules trained on each process must be manually separated by setting "molecule_list", and each process contains the same number of data files. Each process will output a loss file (loss_0,1,2,3.txt) including the MAE loss for data within the process. The total loss is then the averate of values in the four files.

3.2 Demo for using our pre-trained model 

Our pre-trained model "EGNN_hydrocarbon_model.pt" is also included in the repository. In order to use the model to calculate new systems, the model inference script is shown below:

```
from mtelect import infer

device = 'cpu';  # device to run the inference calculation on
scaling = {'V':0.2, 'T': 0.01};  # scaling factors for the neural network output. 
                                 # should be set the same as in the training script
data_path = 'data/aromatic_20_PA_data.json';  # path to the data file of molecule to predict
model_path = 'models/EGNN_hydrocarbon_model.pt';  # path to the pre-trained model
output_path = 'output/';  # path to save the output files

OPS = ['E','x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'yz', 'xz',
       'atomic_charge', 'E_gap', 'bond_order', 'alpha'];      # list of operators to predict

params = {'device':device, 'scaling':scaling, 'data_path':data_path,
          'model_path':model_path, 'OPS':OPS, 'output_path':output_path};

infer(params);
```
Running the inference script by:
```
cd demo
cp demo_inference.py ../
cd ../
python3 demo_inference.py
```
The script reads the molecule data from data/aromatic_20_PA_data.json, which includes the information of the molecule indexed as number 20 in https://pubs.acs.org/doi/10.1021/cr990324%2B. Data files of molecules with index number 19,21,22,23 in the same paper and cyclic polyacetylene ("data/cyclic_PA_data.json") C$_{66}$H$_{66}$ are also included in the demo dataset for users to try. Running the script generates a folder "output/" and write two files "20.json" including all predicted properties and "20_H.json" including the corrected effective Hamiltonian by our EGNN. The predicted properties can be readout by simply loading the json file:
```
import json;

filename = 'output/20.json'
with open(filename,'r') as file:
    data = json.load(file);

data["C"]  # number of carbon atoms
data["H"]  # number of hydrogen atoms
data["E"]["Ehat"][0] # energy of the molecule in Hartree

# electric dipole moment vector reference to the molecule mass center in atomic unit
data["x"]["Ohat"][0]  # px
data["y"]["Ohat"][0]  # py
data["z"]["Ohat"][0]  # pz

# electric quadrupole moment tensor reference to the molecule mass center in atomic unit
data["xx"]["Ohat"][0]  # Qxx
data["yy"]["Ohat"][0]  # Qyy
data["zz"]["Ohat"][0]  # Qzz
data["xy"]["Ohat"][0]  # Qxy
data["xz"]["Ohat"][0]  # Qxz
data["yz"]["Ohat"][0]  # Qyz

# list of Mulliken atomic charges in atomic unit (length = number of atoms)
data["atomic_charge"]["Chat"][0]  # charge of the atom
# 2D nested list of Mayer bond orders (number of atoms x number of atoms)
data["bond_order"]["Bhat"][0]  # bond order between atom i and atom j

data["E_gap"]["Eg_hat"][0]  # optical gap in Hartree
data["alpha"]["alpha_hat"][0]  # static electric polarizability in atomic unit (3x3 nested list)

```

4. Instructions for use

4.1 model training with large dataset

In our paper, the model is trained on a much larger training dataset than the demo case. The dataset includes about 5 times more molecules and about 500 times more configurations. Each molecule has about 100 vibrational configurations, so that the model can capture electronic interaction when the system deviates from equilibrium configurations. However, as the filesize exceeds the limit of a Github repository, we cannot put the whole training dataset here. The dataset is available upon reasonable request to authors of the paper (haot@mit.edu, haoweixu@mit.edu, liju@mit.edu).

The large training set contains about 500 atomic configurations for each chemical formula. For example, both CH4_data.json and CH4_obs_mat.json includes 500 different frames. In the demo dataset, data[key] is a list containing only one element; in the full dataset, data[key] is a list containing 500 element, each for one configuration. To conduct such training, consider setting a larger batch_size and N_epoch, for example:
```
batch_size = 100
N_epoch = 1000
```
The training will take about one day with 4 high-performance GPUs working in parallel. 

Users can also create their own dataset for training. We provide data processing script ""script/generate.py" to generate a batch of DFT and CCSD(T) calculations using quantum chemistry software ORCA https://www.orcasoftware.de/tutorials_orca/.  "script/read.py" is used to read the DFT and CCSD(T) calculation results and output the "xxx_data.json" files. "script/generate_obs_mat.py" is then used to generate the "xxx_obs_mat.json" files. Puting all these files into the data/ folder enables training on customized dataset.

This version of the code does not support systems with elements other than carbon and hydrogen yet, but a new version will be available soon that support arbitrary elements defined by user inputs.

4.2 Applying pre-trained model to user-defined system: orca interface

In order to apply a pre-trained EGNN model to a user-defined molecule other than the molecules in our demo data files, user needs to generate the input data file for the EGNN model. This can be realized in two ways. Here we describe the first way, using software ORCA to generate the input. Alternatively, one can use PySCF to generate the input, which will be described in the next section. The advantage of using ORCA to generate input is that  ORCA is a pre-compiled package, so it is fast when calculating large system compared with PySCF. 

Users first need to run a ORCA DFT calculation using the fast-to-evaluate functional BP86 with a medium-sized cc-pVDZ basis set. This calculation is fast for systems up to hundreds of atoms. An example ORCA calculation is shown in "interface/orca/orca_folder/". Atomic structure is defined in the ORCA input script "run.inp", which also contained other DFT parameters (please keep these parameters unchanged. Just replace the configuration into the one you want to calculate). After ORCA is installed, one can launch the ORCA calculation by:
```
cd interface/orca/orca_folder
/path/to/orca/orca run.inp >log
```
The calculation outputs information to the file "log". Then, run the script "interface/orca/read.py":
```
import json;
from mtelect import QM_reader;

path = 'orca_folder/';   # Path of ORCA calculation results
system = 'system';  # Name of the system. Used just in labeling outputs.

reader = QM_reader(path);     
ne = reader.read_ne('');
HF_dic = reader.read_HF('');
matrix_dic = reader.read_matrix('');

output = {};
for key in HF_dic:
    output[key] = [HF_dic[key]];
for key in matrix_dic:
    output[key] = [matrix_dic[key]];
output['name'] = [system];

with open(path + system + '_data.json','w') as file:
    json.dump(output, file);
```
Consider reset system into a name that identify the molecule you want to calculate. The script is launched by 
```
cd ../
python3 read.py
```
An output data file "interface/orca/system_data.json" will then be generated. Moving the data file to the data folder:
```
mv system_data.json ../../data/
```
Then you can use the demo_inference.py script to calculate your system by just replace the data_path line by 
```
data_path = 'data/system_data.json'
```

4.3 Applying pre-trained model to user-defined system: PySCF interface

Available soon ...
