# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:48:51 2023

@author: 17000
"""

import numpy as np;
from sympy.physics.quantum.cg import CG;
from itertools import product;
import scipy;
import torch;

class to_mat():
    
    def __init__(self, device, irreps, J_max = 3):

        self.device = device;
        self.orbs = irreps.get_orbs();
        self.CGdic = {};
        
        self.T_trans = torch.tensor([[1/3,0,0,-1/np.sqrt(3),0,1],
                                     [0,1,0,0,0,0],
                                     [0,0,0,0,1,0],
                                     [0,1,0,0,0,0],
                                     [1/3,0,0,-1/np.sqrt(3),0,-1],
                                     [0,0,1,0,0,0],
                                     [0,0,0,0,1,0],
                                     [0,0,1,0,0,0],
                                     [1/3,0,0,2/np.sqrt(3),0,0]],dtype=torch.float).to(self.device);

        for j1 in range(J_max):
            for j2 in range(J_max):
                Jmin,Jmax = abs(j1-j2),j1+j2;
                JM_list = [];
                
                for J in range(Jmin,Jmax+1):
                    for M in range(-J,J+1):
                        JM_list.append((J,M));
                        
                Mat = np.array([[float(CG(j1,m1,j2,m2,J,M).doit()) 
                        for m1,m2 in product(range(-j1,j1+1), range(-j2,j2+1))] 
                       for (J,M) in JM_list]);
                
                transJ = [self.sh_basis(J) for J in range(Jmin,Jmax+1)];
                Mat_r = scipy.linalg.block_diag(*transJ);
                
                transj1 = self.sh_basis(j1);
                transj2 = self.sh_basis(j2);
                Mat_l   = np.linalg.inv(np.kron(transj1,transj2));
                
                Mat = torch.tensor(np.real(np.dot(Mat_l,np.dot(Mat.T,Mat_r))),dtype=torch.float).to(self.device);
                
                permute = [(2*abs(m1)-(m1>0))*(2*j2+1)+2*abs(m2)-(m2>0) for m1,m2 in product(range(-j1,j1+1), range(-j2,j2+1))] 
                permute = [permute.index(i) for i in range((2*j1+1)*(2*j2+1))];
                
                self.CGdic[(j1,j2)] = Mat[permute];
                
    def sh_basis(self, J: int)-> np.ndarray:
        
        # This function calculate the transformation from
        # real and imaginary spherical harmonic functions for angular momentum J
        # the output is the (2J+1)*(2J+1) transformation matrix.
        
# =============================================================================
#         mat = np.zeros([2*J+1]*2)*1j;
#         for M in range(-J,0):
#             mat[2*abs(M),M+J]=(-1j)**(J+1)/np.sqrt(2);
#             mat[2*abs(M),-M+J]=(-1j)**J/np.sqrt(2);
#         mat[0,J]=(-1j)**J;
#         for M in range(1,J+1):
#             mat[2*abs(M)-1,M+J]=(-1j)**J*(-1)**M/np.sqrt(2);
#             mat[2*abs(M)-1,-M+J]=(-1)**M*(-1j)**(J-1)/np.sqrt(2); 
#         return mat;
# =============================================================================
    
        mat = np.zeros([2*J+1]*2)*1j;
        for M in range(-J,0):
            mat[M+J,M+J]=(-1j)**(J+1)/np.sqrt(2);
            mat[M+J,-M+J]=(-1j)**J/np.sqrt(2);
        mat[J,J]=(-1j)**J;
        for M in range(1,J+1):
            mat[M+J,M+J]=(-1j)**J*(-1)**M/np.sqrt(2);
            mat[M+J,-M+J]=(-1)**M*(-1j)**(J-1)/np.sqrt(2); 
        return mat;
    
    def to_mat(self, input_tensor:torch.Tensor, j1:int, j2:int, c1:int, c2:int)-> torch.Tensor:
        
        # This function take a tensor c1*c2*(Irreps('|j1-j2|+...+(j1+j2)'))
        # as input and output a batch of rank-2 tensors c1*Irreps(j1)xc2*Irreps(j2)
        # j1,2 are angular momentum and c1,2 are the number of channels 
        # rank-3 tensor: input_tensor[i,j,k], i goes through nodes/edges,
        # j goes though c1*c2 channels, k goes through indices of the Irreps
        # rank-3 tensor output: output[i,m1,m2], m1,2 goes through c1,2x(2j1,2+1)
        
        Mat = self.CGdic[(j1,j2)];
        t_in = input_tensor.reshape([-1,c1,c2,(2*j1+1)*(2*j2+1)]);
        t_out = torch.einsum('ij,uklj->ukli',[Mat,t_in]);
        t_out = t_out.reshape([-1,c1,c2,2*j1+1,2*j2+1]);
        t_out = t_out.permute([0,1,3,2,4]);
        
        return t_out.reshape([-1,c1*(2*j1+1),c2*(2*j2+1)]);
    
    
    def transform(self, V:torch.Tensor, e1, e2)-> torch.Tensor:
        
        # Transform a batch of self.irreps_out tensor into a batch of
        # Irreps('3x0e+2x1o+1x2e')^2 rank-2 tensors
        # Input: V[i,j], i goes through N_node/N_edge; j goes through 14^2=169
        # Output: V[i,m1,m2], m1,m2 goes through len('3x0e+2x1o+1x2e')=14
        
        orb1, orb2 = self.orbs[e1], self.orbs[e2];
        J1, J2 = len(orb1)-1, len(orb2)-1;
        num1 = [orb1[i]*(2*i+1) for i in range(len(orb1))];
        num2 = [orb2[i]*(2*i+1) for i in range(len(orb2))];
        
        l = [i*j for i,j in product(num1,num2)];
        l = [sum(l[:i]) for i in range(len(l)+1)];
        
        Varray = [];
        i_ind = 0;
        for u in range(J1+1):
            Varray.append([]);
            for v in range(J2+1):
                Varray[u].append(self.to_mat(V[:,l[i_ind]:l[i_ind+1]],u,v,orb1[u],orb2[v]))
                i_ind += 1;
    
        v = [torch.cat([Vu for Vu in Varray[u]],dim=2) for u in range(J1+1)];
        
        return torch.cat(v,dim=1);
    
    def T_mat(self, screen, natm, batch):

        self.natm_mat = (batch[None, :] == torch.tensor(list(range(len(natm)))).to(self.device)[:, None]).float();
                               
        Tmat = torch.einsum('ai,kl,il->ak', 
                [self.natm_mat, self.T_trans, screen.reshape([sum(natm),6])]);

        Tmat = Tmat.reshape([len(natm),3,3])/torch.tensor(natm).to(self.device)[:,None,None]**2;
        return Tmat;

    def G_mat(self, gap_mat, natm, batch):

        w = torch.exp(gap_mat[:,:1])
        ave = torch.matmul(self.natm_mat, w * gap_mat[:, 1:]);
        ave = ave / torch.matmul(self.natm_mat, w);

        return ave;

    def raw_to_mat(self, V_raw, minibatch):
        
        nodes = [];
        edges = [];
        for i in range(len(self.orbs)):
            nodes.append(self.transform(V_raw['node'][i],i,i));

        ind = 0;
        for i in range(len(self.orbs)):
            for j in range(i+1):
                edges.append(self.transform(V_raw['edge'][ind],i,j));
                ind += 1;

        screen = V_raw['screen'];
        gap = V_raw['gap'];
        norbs = minibatch['norbs'];
        map1 = minibatch['map1'];
        num_nodes = minibatch['num_nodes'];
        batch = minibatch['batch'];
        natm = minibatch['natm'];
        f_in = minibatch['f_in'];
        pair_ind = minibatch['pair_ind'];
        edge_src = minibatch['edge_src'];
        edge_dst = minibatch['edge_dst'];

        Vmat = [torch.zeros([max(norbs), max(norbs)], dtype=torch.float).to(self.device) for orbs in norbs];
        Tmat = self.T_mat(screen, natm, batch);
        
        for i in range(num_nodes):
            frame = batch[i];
            v = i - sum(natm[:frame]);
            ind = torch.argwhere(f_in[i]).reshape(-1);
            Vmat[frame][map1[frame][v]:map1[frame][v+1], 
                        map1[frame][v]:map1[frame][v+1]] = nodes[ind][i];

        for i, IJind in enumerate(pair_ind):
            for j,ind in enumerate(IJind):
                u1,u2 = edge_src[ind],edge_dst[ind];
                frame = batch[u1];
                v1 = u1 - sum(natm[:frame]);
                v2 = u2 - sum(natm[:frame]);
                Vmat[frame][map1[frame][v1]:map1[frame][v1+1], 
                            map1[frame][v2]:map1[frame][v2+1]] = edges[i][j];
        
        Vmat = torch.stack([(V+V.T)/2 for V in Vmat]);
        gap = self.G_mat(gap, natm, batch);

        return Vmat, Tmat, gap;
        
        