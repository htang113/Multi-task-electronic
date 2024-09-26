import numpy as np;
import json;
import os;

route = os.getcwd()+'/';
molecule_list = ['methane','ethane','acetylene','ethylene',
                 'propane','propylene','cyclopropane','propyne',
                 'cyclobutane','cyclopentane','butane','neopentane',
                 '2m2b','benzene','isobutane','isobutylene',
                 '1butyne','2butyne','13butadiene','12butadiene'];

res = {};
for molecule in molecule_list:
    print('collecting '+molecule+' data');
    res[molecule] = {'E':[],'T':[]};

    for i in range(100):
        print(i)        
        data = os.popen("grep -A 7 'ABSORPTION SPECTRUM' "+route+molecule+'/eom/'+str(i)+'/log').readlines();
        data = [[float(v) for v in u.split()] for u in data[-3:]];
        res[molecule]['E'].append([u[1] for u in data]);
        res[molecule]['T'].append([[u[5],u[6],u[7]] for u in data]);

with open('eom.json','w') as file:
    json.dump(res,file);

