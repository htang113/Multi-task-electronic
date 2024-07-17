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
