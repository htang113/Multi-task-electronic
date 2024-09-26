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
