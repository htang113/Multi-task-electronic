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
