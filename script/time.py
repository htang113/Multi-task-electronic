import os;
import numpy as np;
import json;

molecule_list = ['isobutane','propane','methane','butane','propylene','acetylene',
                 'cyclobutane','ethane','ethylene','benzene','cyclopropane','propyne',
                 'isobutylene','cyclopentane','neopentane','2m2b',
                 '1butyne','2butyne','12butadiene','13butadiene'];

output = {};

for molecule in molecule_list:
    print(molecule)
    HF = [];
    CC = [];

    for i in range(500):
        try:
            data = os.popen('tail '+molecule+'/pvdz/'+str(i)+'/log').readlines()[-1];
            res = data[:-1].split();        
            time = float(res[-6])*60+float(res[-4])+float(res[-2])/1000;
            HF.append(time);
            data = os.popen('tail '+molecule+'/pvtz/'+str(i)+'/log').readlines()[-1];
            res = data[:-1].split();
        
            time = float(res[-6])*60+float(res[-4])+float(res[-2])/1000;
            CC.append(time);
        except:
            print(str(i)+' fail')
    data = os.popen('tail '+molecule+'/b3lyp/log').readlines()[-1];
    res = data[:-1].split();
    time = float(res[-6])*60+float(res[-4])+float(res[-2])/1000;
    
    HFmean = np.mean(HF);
    CCmean = np.mean(CC);
    output[molecule] = {'HF':HFmean, 'CC':CCmean, 'B3LYP':time};

    with open('time.json','w') as file:
        json.dump(output, file);

