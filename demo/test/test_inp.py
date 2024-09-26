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
    'bond_order':0.02, 'alpha':3E-5};

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