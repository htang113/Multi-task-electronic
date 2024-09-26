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
    'bond_order':0.02, 'alpha':3E-5};

params['device'] = 'cuda:0';             # device to run the code for serial training. 
                                         # set as 'cpu' for cpu training and 'cuda:0' for gpu training
params['batch_size'] = 400;              # batch size for training
params['steps_per_epoch'] = 1;           # number of training steps per epoch
params['N_epoch'] = 2001;                # number of epochs for training
params['lr_init'] = 5E-3;                # initial learning rate
params['lr_final'] = 1E-4;               # final learning rate
params['lr_decay_steps'] = 100;          # number of steps for learning rate decay
params['scaling'] = {'V':1, 'T': 0.01};  # scaling factors for the neural network correction terms. 
                                         # V: Hamiltonian correction, T: screening matrix for polarizability.
                                         # should the same as the scaling factors used in the model training.
params['Nsave'] = 100;                   # number of epochs to save the model

params['element_list'] = ['H','C','N','O','F'];           # list of elements in the dataset
                                                          # should be the same as the element_list used in the model training
params['path'] = os.getcwd() +'/';                        # path to the package directory
params['datagroup'] = ['group'+str(i) for i in range(8)]; # list of data groups for training
params['load_model'] = True;                              # load the pre-trained model
params['world_size'] = 16;                                # number of GPUs for parallel training
params['output_path']= os.getcwd()+'/output/';            # output path for the training results
params['ddp_mode'] = 'elastic';                           # distributed data parallel mode for training

if(__name__ == '__main__' or params['ddp_mode'] == 'elastic'):

    main(params);  # train the model


