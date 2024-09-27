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
