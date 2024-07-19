import numpy as np;
import os;
import json;
import scipy;

path = 'orca_folder/';
system = 'system';
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
