# Multi-task-electronic
This package provides a python realization of the multi-task EGNN (equivariant graph neural network) for molecular electronic structure described in the paper "Multi-task learning for molecular electronic structure approaching coupled-cluster accuracy".

# 1. System requirements

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
periodictable==1.7.0
pyscf==2.6.2

nvidia-dali-cuda120==1.35.0  (required for using GPU in the calculation)

Note that in most cases, different version of packages should also work. We list exactly the versions in our calculations in case version inconsistency issue occurs. If users intend to run the program on a cpu device, the cuda package is not needed.

# 2. Installation

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
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install --upgrade e3nn
pip install sympy
pip install periodictable
pip install --prefer-binary pyscf
```
After installing the dependent packages, download this code package:
```
git clone https://github.com/htang113/ML_electronic/
```
Finally enter the working folder and install the package:
```
cd ML_electronic
pip install .
```

# 3. Demo

We include 6 demo scripts in demo/ for training and testing the EGNN model and using a pre-trained model to calculate molecular properties. 

## 3.1 Demo for training a model

### 3.1.1 Serial model training 

The training demo/train_serial/demo_train.py script is shown below:
```
from train import main;
import os;
# This script is used to train the multi-task electronic structure model
params = {};
# properties and their weights in the loss function in the training
# V: regularization term, E: energy,  x,y,z: electric dipole, 
# xx,yy,zz,xy,yz,xz: electric quadrupole, atomic_charge: atomic charge,
# E_gap: optical gap, bond_order: bond order, alpha: electric static polarizability 
params['OPS'] = {'V':0.01,'E':1,
    'x':0.1, 'y':0.1, 'z':0.1,
    'xx':0.01, 'yy':0.01, 'zz':0.01,
    'xy':0.01, 'yz':0.01, 'xz':0.01,
    'atomic_charge': 0.01, 'E_gap':0.2,
    'bond_order':0.02, 'alpha':3E-5, 'F':0.1};
params['device'] = 'cuda:0';             # device to run the code for serial training. 
                                         # set as 'cpu' for cpu training and 'cuda:0' for gpu training
params['batch_size'] = 20;              # batch size for training
params['steps_per_epoch'] = 1;           # number of training steps per epoch
params['N_epoch'] = 41;                # number of epochs for training
params['lr_init'] = 5E-3;                # initial learning rate
params['lr_final'] = 1E-4;               # final learning rate
params['lr_decay_steps'] = 10;          # number of steps for learning rate decay
params['scaling'] = {'V':1, 'T': 0.01};  # scaling factors for the neural network correction terms. 
                                         # V: Hamiltonian correction, T: screening matrix for polarizability.
                                         # should the same as the scaling factors used in the model training.
params['Nsave'] = 10;                   # number of epochs to save the model
params['element_list'] = ['H','C','N','O','F'];           # list of elements in the dataset
                                                          # should be the same as the element_list used in the model training
params['path'] = os.getcwd() +'/';                        # path to the package directory
params['datagroup'] = ['group_train']; # list of data groups for training
params['load_model'] = False;                             # load the pre-trained model
params['world_size'] = 1;                                 # number of GPUs for parallel training
params['output_path']= os.getcwd()+'/output/';            # output path for the training results
params['ddp_mode'] = 'serial';                            # distributed data parallel mode for training
                                                          # set as 'serial','spawn','elastic' for serial, multiprocessing spawn, and elastic ddp modes
if(__name__ == '__main__' or params['ddp_mode'] == 'elastic'):
    main(params);
```
A small training dataset is in the "dataset/group_train/" folder, including starting-point DFT Hamiltonian and CCSD(T) labels of 60 molecules in the QM9 database at equilibrium configuration. The above script can be launched by the following commands:
```
cd demo/train_serial/
cp train_inp.py ../../
cd ../../
python3 train_inp.py
```
in the repository folder. The training takes ~10 minutes on a normal Desktop computer. Running the program writes the loss data into the output/loss/loss_0.txt file and output the trained model named "10_model.pt", ..., "40_model.pt" saved as checkpoints at the 10, 20, 30, and 40th epoch, respectively. The loss data looks as below, including the mean square error loss of all trained quantities (V: the $\parallel V_\theta\parallel^2$ regularization, E: energy, x/y/z:different components of electric dipole moments, xx/yy/zz/xy/yz/xz: different components of electric quadrupole moments, atomic_charge: Mulliken atomic charge, bond_order: Mayer bond order, alpha: static electric polarizability, F: forces on nuclei)

![image](https://github.com/user-attachments/assets/9404599e-da94-4312-bda7-5ae88ca61d23)

Depending on the random seed, specific numbers can be different, but the decreasing trend of the training loss is expected in all cases.

### 3.1.2 Model training with distributed data parallel (DDP) 

The training can also be implemented on multiple GPU's or multiple nodes in parallel using the distributed data parallel (DDP) scheme in "demo/train_multigpus/train_inp.py" and "demo/train_multinodes/train_inp.py". The prior uses multiprocessing spawn to launch the DDP training on 4 GPUs in a single node, and the later uses elastic DDP scheme to train on 16 GPUs in 4 nodes. The DDP mode is switched on by simply setting:
```
params['world_size'] = 4;                                 # number of GPUs for parallel training
params['ddp_mode'] = 'spawn';                            # distributed data parallel mode for training
```
for the first case. Launching the calculation in the same way as the serial version, the script will implement the same training on 4 GPU's in parallel. This calculation takes about 5 min on the NERSC perlmutter GPU node with 4 Nvidia A100 40 GB GPUs https://docs.nersc.gov/systems/perlmutter/architecture/. If your device has a different number of GPUs, change "world_size" accordingly. Note that in the current version, molecules trained on each process are separated automatically, and each process contains rawly the same number of molecules. Each process will output a loss file (loss_0,1,2,3.txt) into output/loss/ folder including the MAE loss for data within the process. The total loss is then the averate of values in the four files.

In the second case, we show how to conduct large scale training. The input script is changed as below:
```
params['world_size'] = 16;                                 # number of GPUs for parallel training
params['ddp_mode'] = 'elastic';                            # distributed data parallel mode for training
params['datagroup'] = ['group'+str(i) for i in range(10)]; # list of data groups for training.
                                                           # Provide training data in dataset/group0,1,2,.../ accordingly
```
together with a separate script demo/train_multinodes/torchrun_script.sh:
```
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export OMP_NUM_THREADS=1
torchrun --nnodes=4 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train_inp.py
```
Such training is usually submitted to high-performance computing systems. We provide an example script demo/train_multinodes/submit.sh for slurm submission on NERSC Perlmutter, in which the training is launched by command:
```
demo/train_multinodes/torchrun_script.sh
```
The training is launched by the following command:
```
cd demo/train_multinodes/
cp train_inp.py torchrun_script.sh submit.sh ../../
cd ../../
sbatch submit.sh
```

## 3.2 Demo for testing a pre-trained model 

Our pre-trained model "output/model/QM9_model.pt" is also included in the repository. To test the model performance on a molecule dataset with both starting-point information and coupled-cluster results, script "demo/test/test_inp.py" is shown below:
```
from deploy import test_func;
import os;
# This script is used to test a pre-trained multi-task electronic structure model
params = {};
# properties to be tested.
# V: regularization term, E: energy,  x,y,z: electric dipole, 
# xx,yy,zz,xy,yz,xz: electric quadrupole, atomic_charge: atomic charge,
# E_gap: optical gap, bond_order: bond order, alpha: electric static polarizability 
# The weights are not used in the testing
params['OPS'] = {'E':1,
    'x':0.1, 'y':0.1, 'z':0.1,
    'xx':0.01, 'yy':0.01, 'zz':0.01,
    'xy':0.01, 'yz':0.01, 'xz':0.01,
    'atomic_charge': 0.01, 'E_gap':0.2,
    'bond_order':0.02, 'alpha':3E-5, 'F':0.1};
params['device'] = 'cuda:0';  # device to run the code for calculations in the testing. 
                              # set as 'cpu' for cpu testing and 'cuda:0' for gpu testing
params['batch_size'] = 5;    # batch size for testing
params['scaling'] = {'V':1, 'T': 0.01};         # scaling factors for the neural network correction terms.
                                                # V: Hamiltonian correction, T: screening matrix for polarizability.
                                                # should the same as the scaling factors used in the model training.
params['element_list'] = ['H','C','N','O','F']; # list of elements in the dataset
                                                # should be the same as the element_list used in the model training
params['path'] = os.getcwd() +'/';              # path to the package directory
params['datagroup'] = ['group_test']; # list of data groups for testing
params['output_path']= os.getcwd()+'/output/';                # output path for the testing results
params['model_file'] = 'QM9_model.pt';                       # model file for testing
test_func(params);  # test the model
```
The code test the model by example data in folder "dataset/group_test/". Implement the testing by commands below:
```
cd demo/test/
cp test_inp.py ../../
cd ../../
python3 test_inp.py
```
The calculation takes about 1 minutes in a normal computer. The results are output to "output/test/test.json". The json file includes a dictionary with keys 'E' (energy, hartree),'x', 'y', 'z' (- electronic dipole moment a.u.), 'xx', 'yy', 'zz', 'xy', 'yz', 'xz' (- electronic quadrupole moment to mass center a.u.), 'atomic_charge' (electron population, e), 'E_gap' (verticle optical gap, hartree), 'bond_order' (bond order, [(atom index 1, atom index 2, bond order), ...]), 'alpha' (static electric polarizability). Each key corresponds to a value {'pred':[...], 'label':[...]}, where 'pred' and 'label' includes lists of the predicted results and coupled-cluster results, respectively. The root-mean-square error can then be straight-forwardly calculated from the output test.json file.

## 3.3 Demo for using a pre-trained model to predict new molecules

We provide interfaces with electronic structure code ORCA and PySCF to use the model in predictions of user-defined molecules. The most convenient way to use the model is by the built-in PySCF interface (3.3.2), which we recommend if the studied molecule contains less than 100 atoms. If user wants to evaluate very large molecules or evaluate one molecule repeatedly with different models, we recommend the ORCA interface, where users prepare starting-point DFT data into files and call the model to do neural network corrections.

### 3.3.1 Inference interface with ORCA

In order to use the model to calculate new systems, the model inference script "demo/infer_orca/infer_inp.py" is shown below:

```
from deploy import infer_func;
import os;
# This script is used to apply a pre-trained multi-task electronic structure model to infer properties of molecules
params = {};
# properties to be inferred.
# V: regularization term, E: energy,  x,y,z: electric dipole,
# xx,yy,zz,xy,yz,xz: electric quadrupole, atomic_charge: atomic charge,
# E_gap: optical gap, bond_order: bond order, alpha: electric static polarizability
# The weights are not used in the inference
params['OPS'] = {'E':1,
    'x':0.1, 'y':0.1, 'z':0.1,
    'xx':0.01, 'yy':0.01, 'zz':0.01,
    'xy':0.01, 'yz':0.01, 'xz':0.01,
    'atomic_charge': 0.01, 'E_gap':0.2,
    'bond_order':0.02, 'alpha':3E-5};
params['device'] = 'cuda:0'; # device to run the code for calculations in the inference.
params['batch_size'] = 4;    # batch size for inference
params['scaling'] = {'V':1, 'T': 0.01};         # scaling factors for the neural network correction terms.
                                                # V: Hamiltonian correction, T: screening matrix for polarizability.
                                                # should the same as the scaling factors used in the model training.
params['element_list'] = ['H','C','N','O','F']; # list of elements in the dataset
                                                # should be the same as the element_list used in the model training
params['path'] = os.getcwd() +'/';              # path to the package directory
params['datagroup'] = ['group_infer'];          # list of data groups for inference
params['output_path']= os.getcwd()+'/output/';  # output path for the inference results
params['model_file'] = 'QM9_model.pt';         # model file for inference
infer_func(params); # infer the properties of molecules
```
Running the inference script by:
```
cd demo/infer_orca
cp infer_inp.py ../../
cd ../../
python3 infer_inp.py
```
The script reads the molecule data from "dataset/group_infer/basic/", which includes the starting-point information of 4 example molecules in the QM9 database. Running the script writes all predicted properties by our EGNN to folder output/inference/inference.json. The predicted properties can be readout by simply loading the json file:
```
import json;

filename = 'output/inference/inference.json'
with open(filename,'r') as file:
    data = json.load(file);
i = 0      # the index (ith) of molecule in the calculation
data["name"][i] # name of the molecule file 
data["E"][i]  # total energy of the molecule in kcal/mol

# electric dipole moment vector reference to the molecule mass center in atomic unit
data["x"][i]  # px
data["y"][i]  # py
data["z"][i]  # pz

# electric quadrupole moment tensor reference to the molecule mass center in atomic unit
data["xx"][i]  # Qxx
data["yy"][i]  # Qyy
data["zz"][i]  # Qzz
data["xy"][i]  # Qxy
data["xz"][i]  # Qxz
data["yz"][i]  # Qyz

# list of Mulliken atomic charges in atomic unit (length = number of atoms)
data["atomic_charge"][i]  # charge of the atom
# 2D nested list of Mayer bond orders (number of bonds x 3)
data["bond_order"][i]  # bond order Bij between atom i and atom j as a 3-elements tuple: (i, j, Bij)

data["E_gap"][i]  # optical gap in eV
data["alpha"][i]  # static electric polarizability in atomic unit (3x3 nested list)
data["F"][i]      # Forces on nuclei in atomic unit (3xN nested list)
```
Note that the local DFT starting point from ORCA is already provided in this demo. If the user want to use the ORCA interface to study new molecules, it is necessary to implement ORCA DFT calculations and prepare the data file in the same format as "dataset/group_infer/basic/". Detailed instructions on how to prepare the data is elaborated in section 4.2.

### 3.3.2 Inference interface with PySCF
The model inference script interface with PySCF "demo/infer_pyscf/pyscf_inp.py" is shown below:
```
from deploy import pyscf_func;
import os;
# This script is used to apply a pre-trained multi-task electronic structure model to infer properties of molecules
params = {};
params['device'] = 'cuda:0'; # device to run the code for calculations in the inference.
params['scaling'] = {'V':1, 'T': 0.01};         # scaling factors for the neural network correction terms.
                                                # V: Hamiltonian correction, T: screening matrix for polarizability.
                                                # should the same as the scaling factors used in the model training.
params['element_list'] = ['H','C','N','O','F']; # list of elements in the dataset
                                                # should be the same as the element_list used in the model training
params['path'] = os.getcwd() +'/';              # path to the package directory
params['output_path']= os.getcwd()+'/output/';  # output path for the inference results
params['model_file'] = 'QM9_model.pt';         # model file for inference
elements = ['C','H','H','H','H'];         # list of elements in the molecule
pos = [[0.000000, 0.000000, 0.000000],
       [0.000000, 0.000000, 1.089000],
       [1.026719, 0.000000, -0.363000],
       [-0.513360, -0.889165, -0.363000],
       [-0.513360, 0.889165, -0.363000]]; # list of atomic positions in the molecule (Angstrom)
name = 'methane';                         # name of the molecule
properties = pyscf_func(params, elements, pos, name); # infer the properties of molecules
```
The script calculate properties of a methane molecule whose atomic species and coordinates are directly defined in the script as "elements" and "pos". The function pyscf_func takes related settings and molecular properties as input and output the predicted properties into file 'output/inference/methane.json' in the same format as the orca inference output. Note that all input information for the PySCF script is atomic species and coordinates, and the starting point DFT will be calculated using PySCF within our package.

# 4. Instructions for use

## 4.1 model training with large dataset

In our paper, the model is trained on a much larger training dataset than the demo case. The dataset includes about 5 times more molecules and about 500 times more configurations. Each molecule has about 100 vibrational configurations, so that the model can capture electronic interaction when the system deviates from equilibrium configurations. However, as the filesize exceeds the limit of a Github repository, we cannot put the whole training dataset here. The dataset is available upon reasonable request to authors of the paper (haot@mit.edu, wenhaohe@mit.edu, haoweixu@mit.edu, liju@mit.edu).

The large training set contains about 500 atomic configurations for each chemical formula. For example, both CH4_data.json and CH4_obs_mat.json includes 500 different frames. In the demo dataset, data[key] is a list containing only one element; in the full dataset, data[key] is a list containing 500 element, each for one configuration. To conduct such training, consider setting a larger batch_size and N_epoch, for example:
```
batch_size = 100
N_epoch = 1000
```
The training will take about one day with 4 high-performance GPUs working in parallel. 

Users can also create their own dataset for training. We provide data processing script ""script/generate.py" to generate a batch of DFT and CCSD(T) calculations using quantum chemistry software ORCA https://www.orcasoftware.de/tutorials_orca/.  "script/read.py" is used to read the DFT and CCSD(T) calculation results and output the "xxx_data.json" files. "script/generate_obs_mat.py" is then used to generate the "xxx_obs_mat.json" files. Puting all these files into the data/ folder enables training on customized dataset.

This version of the code does not support systems with elements other than carbon and hydrogen yet, but a new version will be available soon that support arbitrary elements defined by user inputs.

## 4.2 Applying pre-trained model to user-defined system: orca interface

In order to apply a pre-trained EGNN model to a user-defined molecule other than the molecules in our demo data files, user needs to generate the input data file for the EGNN model. This can be realized in two ways. Here we describe the first way, using software ORCA to generate the input. Alternatively, one can use PySCF to generate the input, which will be described in the next section. The advantage of using ORCA to generate input is that  ORCA is a pre-compiled package, so it is fast when calculating large system compared with PySCF. 

Users first need to run a ORCA DFT calculation using the fast-to-evaluate functional BP86 with a medium-sized def2-SVP basis set. This calculation is fast for systems up to hundreds of atoms. An example ORCA calculation is shown in "demo/infer_orca/orca_folder". Atomic structure is defined in the ORCA input script "run.inp", which also contained other DFT parameters (please keep these parameters unchanged. Just replace the configuration into the one you want to calculate). After ORCA is installed, one can launch the ORCA calculation by:
```
cd demo/infer_orca/orca_folder
/path/to/orca/orca run.inp >log
```
One can also utilize the parallel computing feature of ORCA and submit the calculation to slurm system. Please see ORCA tutorials (https://www.faccts.de/docs/orca/5.0/tutorials/, https://sites.google.com/site/orcainputlibrary/) for efficient ways to implement large DFT calculations in the software. The calculation outputs information to the file "log". Then, run the script "demo/infer_orca/read.py":
```
import json;
from utils import QM_reader;

path = 'orca_folder/';   # Path of ORCA calculation results
system = 'example';      # Name of the system. Used just in labeling outputs.

reader = QM_reader(path);     
ne = reader.read_ne('');
HF_dic = reader.read_HF('');
matrix_dic = reader.read_matrix('');

output = {};
for key in HF_dic:
    output[key] = HF_dic[key];
for key in matrix_dic:
    output[key] = matrix_dic[key];
output['name'] = system;

with open(path + system + '.json','w') as file:
    json.dump(output, file);
```
Consider reset "example" into a name that identify the molecule you want to calculate. The script is launched by 
```
cd ../
python3 read.py
```
An output data file "demo/infer_orca/orca_folder/example.json" will then be generated. Create a new data group folder and moving the data file to the data folder:
```
mkdir ../../dataset/group_infer_example
mkdir ../../dataset/group_infer_example/basic
cp orca_folder/example.json ../../dataset/group_infer_example/basic/
```
Then you can use the infer_inp.py script to calculate your system by just replace the data_path line by 
```
params['datagroup'] = ['group_infer_example'];
```

