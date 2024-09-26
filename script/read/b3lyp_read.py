import os;
import numpy as np;
import json;

def read_obs(folder, natom):
    obs_dic = {};
    dipole = os.popen("grep -A 4 'Electronic Contribution' "+folder+'/run_property.txt');
    dipole = [-float(u[:-1].split()[-1]) for u in dipole.readlines()[-3:]];
    
    obs_dic['x'] = dipole[0];
    obs_dic['y'] = dipole[1];
    obs_dic['z'] = dipole[2];
    
    quadrupole = os.popen("grep -A 4 'Electronic part' "+folder+'/run_property.txt');
    quadrupole = [[-float(v) for v in u[:-1].split()[1:]] for u in quadrupole.readlines()[-3:]];
    
    obs_dic['xx'] = quadrupole[0][0];
    obs_dic['yy'] = quadrupole[1][1];
    obs_dic['zz'] = quadrupole[2][2];
    obs_dic['xy'] = quadrupole[0][1];
    obs_dic['yz'] = quadrupole[1][2];
    obs_dic['xz'] = quadrupole[0][2];

    command = "grep -A "+str(natom+1)+" 'MULLIKEN ATOMIC CHARGE' ";
    atomicCharge = os.popen(command +folder+'/log').readlines()[-natom:];
    atomicCharge = [float(u[:-1].split()[-1]) for u in atomicCharge];
    obs_dic['atomic_charge'] = atomicCharge;
    
    command = "grep -A "+str(20)+" 'Mayer bond orders' ";
    bond_data = os.popen(command +folder+'/log').readlines();
    i_ind = -1;
    while('Mayer bond orders' not in bond_data[i_ind]):
        i_ind -= 1;
    i_ind += 1;
    bond_order = [];
    while('B' in bond_data[i_ind]):
        u = bond_data[i_ind][:-1].split();
        j_ind = 0;
        while(7*j_ind+7<=len(u)):
            bond_order.append([int(u[7*j_ind+1][:-2]),
                                int(u[7*j_ind+3][:-2]),
                                float(u[7*j_ind+6])
                                ])
            j_ind += 1;
        i_ind += 1;
        
    obs_dic['bond_order'] = bond_order;
    
    return obs_dic;

molecule_list = ['methane','ethane','ethylene','acetylene','propane',
        'propylene','propyne','cyclopropane','butane','isobutane','isobutylene',
        'cyclobutane','cyclopentane','benzene','2m2b','neopentane','1butyne','2butyne',
        '12butadiene','13butadiene'];
natoms = [5, 8, 6, 4, 11, 9, 7, 9, 14, 14,
          12, 12, 15, 12, 15, 17, 10,10,10,10];

route = os.getcwd()+'/';

out = {'CCSD(T)':[],'B3LYP':[],'name':[]};
for i in range(len(molecule_list)):

    molecule = molecule_list[i];
    natom = natoms[i];

    print('calculating '+molecule);

    DFT_dic = read_obs(route+molecule+'/b3lyp', natom);
    CCSD_dic = read_obs(route+molecule+'/pvtz/0', natom);
    HF_dic = {};

    E_DFT = float(os.popen("grep 'DFT Energy' "+route+molecule+'/b3lyp/run_property.txt').readlines()[-1][:-1].split()[-1]);
    E_CCSD = float(os.popen("grep 'E(CCSD(T))' "+route+molecule+'/pvtz/0/log').readlines()[-1][:-1].split()[-1]);
    E_HF = float(os.popen("grep 'SINGLE POINT ENERGY' "+route+molecule+'/pvdz/0/log').readlines()[-1][:-1].split()[-1]);

    HF_dic['E'] = E_HF;
    DFT_dic['E'] = E_DFT;
    CCSD_dic['E'] = E_CCSD;

    out['B3LYP'].append(DFT_dic);
    out['CCSD(T)'].append(CCSD_dic);
    out['name'].append(molecule);

with open('check_DFT.json','w') as file:
    json.dump(out,file);
