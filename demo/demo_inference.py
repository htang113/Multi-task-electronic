# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:00:28 2023

@author: 17000
"""

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
