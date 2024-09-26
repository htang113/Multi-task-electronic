#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:16:53 2023

@author: ubuntu
"""

import json;
import numpy as np;
import scipy;
import os;

def radius_coef(n,z):
    
    res = scipy.special.factorial2(2*n+1)/z**(n+1)/2**(2*n+3)*np.sqrt(np.pi/z/2);
    
    return 1/np.sqrt(res);

def get_orbital(data, line):
    
    ele = data[line-1].split()[-1];
    orbs = [];
    while('end' not in data[line]):
        res = data[line].split();
        zeta, cl, alpha = [], [], [];
        for i in range(int(res[1])):
            line += 1;
            [_, z, c] = data[line].split();
            zeta.append(float(z));
            cl.append(float(c));
        line += 1;
            
        if(res[0]=='S'):
            spherical_harmonic = 1/np.sqrt(4*np.pi);
            radius = np.array([radius_coef(0,z) for z in zeta]);
            alpha = np.zeros([len(zeta),3]).tolist();
            orbs.append({'zeta':zeta, 'c':(np.array(cl)*spherical_harmonic*radius).tolist(), 'alpha': alpha});
        
        elif(res[0]=='P'):
            spherical_harmonic = np.sqrt(3/4/np.pi);
            radius = np.array([radius_coef(1,z) for z in zeta]);
            alpha = [[[0,0,1]]*len(zeta),
                    [[1,0,0]]*len(zeta),
                    [[0,1,0]]*len(zeta)];
            
            for a1 in alpha:
                orbs.append({'zeta':zeta, 'c':(np.array(cl)*spherical_harmonic*radius).tolist(), 'alpha': a1});
                
        elif(res[0]=='D'):
            sh = np.array([np.sqrt(5/np.pi)/4, np.sqrt(15/np.pi)/2,
                                np.sqrt(15/np.pi)/2, np.sqrt(15/np.pi)/4,
                                np.sqrt(15/np.pi)/2]);
            rd = np.array([radius_coef(2,z) for z in zeta]);

            alpha = [[[0,0,2]]*len(zeta)+[[2,0,0]]*len(zeta)+[[0,2,0]]*len(zeta),
                    [[1,0,1]]*len(zeta),
                    [[0,1,1]]*len(zeta),
                    [[2,0,0]]*len(zeta)+[[0,2,0]]*len(zeta),
                    [[1,1,0]]*len(zeta)];
            orbs.append({'zeta':zeta*3, 
                        'c': (np.array(cl)*2*sh[0]*rd).tolist()+(-np.array(cl)*sh[0]*rd).tolist()+(-np.array(cl)*sh[0]*rd).tolist(), 
                        'alpha':alpha[0]});
            
            for j in range(1,3):
                a1 = alpha[j];
                orbs.append({'zeta':zeta, 'c':(np.array(cl)*sh[j]*rd).tolist(), 'alpha': a1});
                
            orbs.append({'zeta':zeta*2, 
                        'c': (np.array(cl)*sh[3]*rd).tolist()+(-np.array(cl)*sh[3]*rd).tolist(), 
                        'alpha':alpha[3]});
            orbs.append({'zeta':zeta, 'c':(np.array(cl)*sh[4]*rd).tolist(), 'alpha': alpha[4]});
        elif(res[0]=='F'):
            sh = np.array([np.sqrt(7/np.pi)/4, np.sqrt(21/2/np.pi)/4,
                                np.sqrt(21/2/np.pi)/4, np.sqrt(105/np.pi)/4,
                                np.sqrt(105/np.pi)/2, np.sqrt(35/2/np.pi)/4,
                                np.sqrt(35/2/np.pi)/4]);
            rd = np.array([radius_coef(3,z) for z in zeta]);

            alpha = [[[0,0,3]]*len(zeta)+[[2,0,1]]*len(zeta)+[[0,2,1]]*len(zeta),
                    [[1,0,2]]*len(zeta)+[[3,0,0]]*len(zeta)+[[1,2,0]]*len(zeta),
                    [[0,1,2]]*len(zeta)+[[2,1,0]]*len(zeta)+[[0,3,0]]*len(zeta),
                    [[2,0,1]]*len(zeta)+[[0,2,1]]*len(zeta),
                    [[1,1,1]]*len(zeta),
                    [[3,0,0]]*len(zeta)+[[1,2,0]]*len(zeta),
                    [[2,1,0]]*len(zeta)+[[0,3,0]]*len(zeta)];
            orbs.append({'zeta':zeta*3, 
                        'c': (np.array(cl)*2*sh[0]*rd).tolist()+(-3*np.array(cl)*sh[0]*rd).tolist()+(-3*np.array(cl)*sh[0]*rd).tolist(), 
                        'alpha':alpha[0]});
            orbs.append({'zeta':zeta*3, 
                        'c': (np.array(cl)*4*sh[1]*rd).tolist()+(-np.array(cl)*sh[1]*rd).tolist()+(-np.array(cl)*sh[1]*rd).tolist(), 
                        'alpha':alpha[1]});
            orbs.append({'zeta':zeta*3, 
                        'c': (np.array(cl)*4*sh[2]*rd).tolist()+(-np.array(cl)*sh[2]*rd).tolist()+(-np.array(cl)*sh[2]*rd).tolist(), 
                        'alpha':alpha[2]});
            orbs.append({'zeta':zeta*2, 
                        'c': (np.array(cl)*sh[3]*rd).tolist()+(-np.array(cl)*sh[3]*rd).tolist(), 
                        'alpha':alpha[3]});
            orbs.append({'zeta':zeta*1, 
                        'c': (np.array(cl)*sh[4]*rd).tolist(), 
                        'alpha':alpha[4]});
            orbs.append({'zeta':zeta*2, 
                        'c': (np.array(cl)*sh[5]*rd).tolist()+(-3*np.array(cl)*sh[5]*rd).tolist(),
                        'alpha':alpha[5]});
            orbs.append({'zeta':zeta*2, 
                'c': (np.array(cl)*3*sh[6]*rd).tolist()+(-np.array(cl)*sh[6]*rd).tolist(), 
                'alpha':alpha[6]});

    return ele, orbs, line;

def read_basis(path, savefile = False):

    if(not os.path.exists(path + '/orbitals_new.json')):

        line = 2;
        output = {};

        with open(path + '/def2-svp','r') as file:
            data = file.readlines();

        while(line < len(data)):
            
            ele, orbs, line = get_orbital(data, line);
            output[ele] = orbs;
            line += 4;

        if(savefile):
            with open(path + '/orbitals_new.json','w') as file:
                json.dump(output,file);
        
        return output;
    
    else:

        return 'file already exists';

