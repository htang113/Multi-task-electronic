# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:00:20 2023

@author: haota
"""

import json;
from pkgs.dataframe import load_data;
from pkgs.integral import integrate;

for task in range(20):
    molecule_list = ['methane','ethane','ethylene','2m2b',
                     'acetylene','propane','propylene','propyne',
                     'cyclopropane','cyclobutane','butane','benzene',
                     'isobutane','isobutylene','neopentane','cyclopentane',
                     '12butadiene','13butadiene','1butyne','2butyne'];
    
    OPS = ['x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'yz', 'xz'];
    device = 'cuda:0';
    molecule_list = [molecule_list[task]];
    
    data, labels, obs_mat = load_data(molecule_list, device, load_obs_mat=False, 
                             ind_list=[], op_names=OPS);
    integrator = integrate(device);
    
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

