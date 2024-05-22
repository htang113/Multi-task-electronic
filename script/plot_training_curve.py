# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:17:57 2023

@author: haota
"""

import numpy as np;
import json;
import matplotlib.pyplot as plt;
import matplotlib;

with open('loss.json','r') as file:
    data = json.load(file);

OPS = {'V':0.1,'E':1,
       'x':0.1, 'y':0.1, 'z':0.1,
       'xx':0.1, 'yy':0.1, 'zz':0.1,
       'xy':0.1, 'yz':0.1, 'xz':0.1}

oplist = list(OPS.keys());

res = {};
for i in range(len(oplist)):
    
    key = oplist[i];
    res[key] = [u[i] for u in data];

plt.figure(figsize=(16,9))

font = {
'size' : 32}

matplotlib.rc('font', **font)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

for key in res:
    if(key!='E'):
        plt.plot(range(len(res[key])), np.log10(res[key]),
                 label=key, linewidth=3);
        
plt.plot(range(len(res['E'])), np.log10(res['E']),
           label='E', linewidth=4,c='black');  
plt.legend(ncol=2)
plt.axis([-3,1000,-5,-0.5]);
plt.xlabel('Epoch');
plt.ylabel('log$_{10}$ loss');
