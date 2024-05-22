#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:16:53 2023

@author: ubuntu
"""

import json;
import numpy as np;
import scipy;

def radius_coef(n,z):
    
    res = scipy.special.factorial2(2*n+1)/z**(n+1)/2**(2*n+3)*np.sqrt(np.pi/z/2);
    
    return 1/np.sqrt(res);
    
with open('basis','r') as file:
    data = file.readlines();

Horbs, Corbs = [], [];

line = 2;
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
        alpha = np.zeros([int(res[1]),3]).tolist();
        Horbs.append({'zeta':zeta, 'c':(np.array(cl)*spherical_harmonic*radius).tolist(), 'alpha': alpha});
    
    if(res[0]=='P'):
        spherical_harmonic = np.sqrt(3/4/np.pi);
        radius = np.array([radius_coef(1,z) for z in zeta]);
        alpha = [[[0,0,1]]*int(res[1]),
                 [[1,0,0]]*int(res[1]),
                 [[0,1,0]]*int(res[1])];
        
        for a1 in alpha:
            Horbs.append({'zeta':zeta, 'c':(np.array(cl)*spherical_harmonic*radius).tolist(), 'alpha': a1});
            
    if(res[0]=='D'):
        sh = np.array([np.sqrt(5/np.pi)/4, np.sqrt(15/np.pi)/2,
                              np.sqrt(15/np.pi)/2, np.sqrt(15/np.pi)/4,
                              np.sqrt(15/np.pi)/2]);
        rd = np.array([radius_coef(2,z) for z in zeta]);

        alpha = [[[0,0,2]]*int(res[1])+[[2,0,0]]*int(res[1])+[[0,2,0]]*int(res[1]),
                 [[1,0,1]]*int(res[1]),
                 [[0,1,1]]*int(res[1]),
                 [[2,0,0]]*int(res[1])+[[0,2,0]]*int(res[1]),
                 [[1,1,0]]*int(res[1])];
        Horbs.append({'zeta':zeta*3, 
                      'c': (np.array(cl)*2*sh[0]*rd).tolist()+(-np.array(cl)*sh[0]*rd).tolist()+(-np.array(cl)*sh[0]*rd).tolist(), 
                      'alpha':alpha[0]});
        
        for j in range(1,3):
            a1 = alpha[j];
            Horbs.append({'zeta':zeta, 'c':(np.array(cl)*sh[j]*rd).tolist(), 'alpha': a1});
            
        Horbs.append({'zeta':zeta*2, 
                      'c': (np.array(cl)*sh[3]*rd).tolist()+(-np.array(cl)*sh[3]*rd).tolist(), 
                      'alpha':alpha[3]});
        Horbs.append({'zeta':zeta, 'c':(np.array(cl)*sh[4]*rd).tolist(), 'alpha': alpha[4]});

line += 4;
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
        alpha = np.zeros([int(res[1]),3]).tolist();
        Corbs.append({'zeta':zeta, 'c':(np.array(cl)*spherical_harmonic*radius).tolist(), 'alpha': alpha});
    
    if(res[0]=='P'):
        spherical_harmonic = np.sqrt(3/4/np.pi);
        radius = np.array([radius_coef(1,z) for z in zeta]);
        alpha = [[[0,0,1]]*int(res[1]),
                 [[1,0,0]]*int(res[1]),
                 [[0,1,0]]*int(res[1])];
        
        for a1 in alpha:
            Corbs.append({'zeta':zeta, 'c':(np.array(cl)*spherical_harmonic*radius).tolist(), 'alpha': a1});
            
    if(res[0]=='D'):
        sh = np.array([np.sqrt(5/np.pi)/4, np.sqrt(15/np.pi)/2,
                              np.sqrt(15/np.pi)/2, np.sqrt(15/np.pi)/4,
                              np.sqrt(15/np.pi)/2]);
        rd = np.array([radius_coef(2,z) for z in zeta]);

        alpha = [[[0,0,2]]*int(res[1])+[[2,0,0]]*int(res[1])+[[0,2,0]]*int(res[1]),
                 [[1,0,1]]*int(res[1]),
                 [[0,1,1]]*int(res[1]),
                 [[2,0,0]]*int(res[1])+[[0,2,0]]*int(res[1]),
                 [[1,1,0]]*int(res[1])];
        Corbs.append({'zeta':zeta*3, 
                      'c': (np.array(cl)*2*sh[0]*rd).tolist()+(-np.array(cl)*sh[0]*rd).tolist()+(-np.array(cl)*sh[0]*rd).tolist(), 
                      'alpha':alpha[0]});
        
        for j in range(1,3):
            a1 = alpha[j];
            Corbs.append({'zeta':zeta, 'c':(np.array(cl)*sh[j]*rd).tolist(), 'alpha': a1});
            
        Corbs.append({'zeta':zeta*2, 
                      'c': (np.array(cl)*sh[3]*rd).tolist()+(-np.array(cl)*sh[3]*rd).tolist(), 
                      'alpha':alpha[3]});
        Corbs.append({'zeta':zeta, 'c':(np.array(cl)*sh[4]*rd).tolist(), 'alpha': alpha[4]});
    
    if(res[0]=='F'):
        sh = np.array([np.sqrt(7/np.pi)/4, np.sqrt(21/2/np.pi)/4,
                              np.sqrt(21/2/np.pi)/4, np.sqrt(105/np.pi)/4,
                              np.sqrt(105/np.pi)/2, np.sqrt(35/2/np.pi)/4,
                              np.sqrt(35/2/np.pi)/4]);
        rd = np.array([radius_coef(3,z) for z in zeta]);

        alpha = [[[0,0,3]]*int(res[1])+[[2,0,1]]*int(res[1])+[[0,2,1]]*int(res[1]),
                 [[1,0,2]]*int(res[1])+[[3,0,0]]*int(res[1])+[[1,2,0]]*int(res[1]),
                 [[0,1,2]]*int(res[1])+[[2,1,0]]*int(res[1])+[[0,3,0]]*int(res[1]),
                 [[2,0,1]]*int(res[1])+[[0,2,1]]*int(res[1]),
                 [[1,1,1]]*int(res[1]),
                 [[3,0,0]]*int(res[1])+[[1,2,0]]*int(res[1]),
                 [[2,1,0]]*int(res[1])+[[0,3,0]]*int(res[1])];
        Corbs.append({'zeta':zeta*3, 
                      'c': (np.array(cl)*2*sh[0]*rd).tolist()+(-3*np.array(cl)*sh[0]*rd).tolist()+(-3*np.array(cl)*sh[0]*rd).tolist(), 
                      'alpha':alpha[0]});
        Corbs.append({'zeta':zeta*3, 
                      'c': (np.array(cl)*4*sh[1]*rd).tolist()+(-np.array(cl)*sh[1]*rd).tolist()+(-np.array(cl)*sh[1]*rd).tolist(), 
                      'alpha':alpha[1]});
        Corbs.append({'zeta':zeta*3, 
                      'c': (np.array(cl)*4*sh[2]*rd).tolist()+(-np.array(cl)*sh[2]*rd).tolist()+(-np.array(cl)*sh[2]*rd).tolist(), 
                      'alpha':alpha[2]});
        Corbs.append({'zeta':zeta*2, 
                      'c': (np.array(cl)*sh[3]*rd).tolist()+(-np.array(cl)*sh[3]*rd).tolist(), 
                      'alpha':alpha[3]});
        Corbs.append({'zeta':zeta*1, 
                      'c': (np.array(cl)*sh[4]*rd).tolist(), 
                      'alpha':alpha[4]});
        Corbs.append({'zeta':zeta*2, 
                      'c': (np.array(cl)*sh[5]*rd).tolist()+(-3*np.array(cl)*sh[5]*rd).tolist(),
                      'alpha':alpha[5]});
        Corbs.append({'zeta':zeta*2, 
              'c': (np.array(cl)*3*sh[6]*rd).tolist()+(-np.array(cl)*sh[6]*rd).tolist(), 
              'alpha':alpha[6]});

output = {'H':Horbs ,'C':Corbs};
with open('orbitals.json','w') as file:
    json.dump(output,file);