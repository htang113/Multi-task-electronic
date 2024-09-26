# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:00:28 2023

@author: 17000
"""
from pyscf import gto, dft
import torch;
import numpy as np
import scipy
from basis.Irreps import Irreps_build;
from basis.integral import integrate;

class generate_basic:

    def __init__(self, device, path, element_list = ['H','C','N','O','F']):

        self.device = device;
        irreps = Irreps_build(element_list);
        self.nbasis = irreps.get_nbasis();
        self.perm_block = {
            's':np.array([[1]]),
            'p':np.array([
                        [0,1,0],
                        [0,0,1],
                        [1,0,0]]),
            'd':np.array([
                        [0,0,0,0,1],
                        [0,0,1,0,0],
                        [1,0,0,0,0],
                        [0,1,0,0,0],
                        [0,0,0,1,0]]),
            }
        irreps = Irreps_build(element_list);
        self.op_names = ['x','y','z','xx','yy','zz','xy','xz','yz'];
        self.integrator = integrate(device, 'def2-SVP', path);
        
    def direct_sum(self, A, B):
        if type(A) == str:
            return B
        if type(B) == str:
            return A
        # Create an output matrix with the appropriate shape filled with zeros
        result = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]))
        
        # Place A in the top-left corner
        result[:A.shape[0], :A.shape[1]] = A
        
        # Place B in the bottom-right corner
        result[A.shape[0]:, A.shape[1]:] = B
        
        return result

    def get_perm_mat(self, mol):

        ind = 0
        perm_mat = 'None'
        while ind < len(mol.ao_labels()):
            l_val = mol.ao_labels()[ind][5]
            if l_val == 's':
                ind += 1
            elif l_val == 'p':
                ind += 3
            elif l_val == 'd':
                ind += 5
            else:
                raise TypeError('wrong l value')
            perm_mat = self.direct_sum(perm_mat, 
                                       self.perm_block[l_val]);

        return perm_mat;

    def get_pyscf_mol(self, elements, pos):

        atom_str = '';
        for ind, ele in enumerate(elements):
            atom_str += ele;
            for x in pos[ind]:
                atom_str += ' '+str(x);
            atom_str += ';';
        atom_str = atom_str[:-1];

        mol = gto.M(atom = atom_str, basis = 'def2-svp', symmetry = True);
    
        return mol

    def read_obs_mat(self, elements, pos, Smhalf):

        data_obs = {}
        for operator in self.op_names:
            res = self.integrator.calc_O(torch.tensor([pos], dtype=torch.float).to(self.device), 
                                         elements, operator)[0];
            output = torch.matmul(torch.matmul(Smhalf,
                                               torch.Tensor(res).to(self.device)),
                                  Smhalf)
            data_obs[operator] = output;

        return data_obs;

    def generate(self, elements, pos, name):

        mol = self.get_pyscf_mol(elements, pos);

        calc = dft.RKS(mol);
        calc.xc = 'bp86';
        calc.kernel();

        h = calc.get_fock();
        S = calc.get_ovlp();
        
        E_nn = mol.energy_nuc()
        ne = mol.tot_electrons()

        h += (-np.sum(scipy.linalg.eigvalsh(h,S)[:int(ne/2)])*2 + calc.e_tot - E_nn)/ne*S;
        
        Smhalf = torch.tensor(scipy.linalg.fractional_matrix_power(S, (-1/2)),
                                dtype=torch.float).to(self.device);
        S = torch.tensor(S, dtype=torch.float).to(self.device);
        h = torch.tensor(h, dtype=torch.float).to(self.device);
        h = torch.matmul(torch.matmul(Smhalf, h),Smhalf);
        perm_mat = torch.tensor(self.get_perm_mat(mol), 
                                dtype=torch.float).to(self.device);
        pos = np.array(pos)[:,[1,2,0]].tolist();

        data_in = {}
        data_in['elements'] = elements;
        data_in['pos'] = torch.tensor(pos, dtype=torch.float).to(self.device);
        data_in['h'] = perm_mat.T @ h @ perm_mat
        data_in['S'] = perm_mat.T @ S @ perm_mat
        data_in['Smhalf'] = perm_mat.T @ Smhalf @ perm_mat
        data_in['E_nn'] = E_nn
        data_in['ne'] = int(round(ne/2))
        data_in['norbs'] = int(round(sum([self.nbasis[ele] for ele in elements])));
        data_in['name'] = name;
        data_obs = self.read_obs_mat(elements, pos, data_in['Smhalf']);

        return data_in, data_obs;
