# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:00:20 2023

@author: haota
"""

import json;
from data import load_data;
from basis import integrate;

starting_basis = 'cc-pVDZ';

for task in range(23):

    molecule_list = ['CH4','C2H2','C2H4','C2H6','C3H4',
                 'C3H6','C3H8','C4H6','C4H8','C4H10',
                 'C5H8','C5H10','C5H12','C6H6','C6H8',
                 'C6H12','C6H14','C7H8','C7H10','C8H8'];
    molecule_list += ['C7H14', 'C8H14', 'C10H10'];
    OPS = ['x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'yz', 'xz'];
    device = 'cuda:0';
    molecule_list = [molecule_list[task]];
    
    data, labels, obs_mat = load_data(molecule_list, device, load_obs_mat=False, 
                             ind_list=[], op_names=OPS);
    integrator = integrate(device, starting_basis=starting_basis);
    
    pos = data[0]['pos'];
    atm = data[0]['elements'];
    
    obs_mat = {};
    
    for operator in OPS:
        print('calculating '+str(task)+' '+operator);
        
        res = [];
        res += integrator.calc_O(pos,atm,operator);
        obs_mat[operator] = res;
        
    
    with open('data/'+molecule_list[0]+'_obs_mat.json', 'w') as file:
        
        json.dump(obs_mat, file);

