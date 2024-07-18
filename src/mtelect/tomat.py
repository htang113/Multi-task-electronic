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
    
    def __init__(self, device, J_max = 3):

        self.device = device;
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
    
    
    def transform(self, V:torch.Tensor, J1, J2)-> torch.Tensor:
        
        # Transform a batch of self.irreps_out tensor into a batch of
        # Irreps('3x0e+2x1o+1x2e')^2 rank-2 tensors
        # Input: V[i,j], i goes through N_node/N_edge; j goes through 14^2=169
        # Output: V[i,m1,m2], m1,m2 goes through len('3x0e+2x1o+1x2e')=14
        
        if(J1==1 and J2==1):
            
            num1, num2 = [2,3], [2,3];
        
        elif(J1==2 and J2 == 1):
            num1, num2 = [3,6,5], [2,3];
        
        elif(J1==2 and J2==2):
            
            num1, num2 = [3,6,5], [3,6,5];
        
        elif(J1==3 and J2==3):
            
            num1, num2 = [4,9,10,7], [4,9,10,7];
        
        elif(J1==3 and J2==2):
            
            num1, num2 = [4,9,10,7], [3,6,5];
            
        l = [i*j for i,j in product(num1,num2)];
        l = [sum(l[:i]) for i in range(len(l)+1)];
        
        Varray = [];
        i_ind = 0;
        for u in range(J1+1):
            Varray.append([]);
            for v in range(J2+1):
                Varray[u].append(self.to_mat(V[:,l[i_ind]:l[i_ind+1]],u,v,J1+1-u,J2+1-v))
                i_ind += 1;
    
        v = [torch.cat([Vu for Vu in Varray[u]],dim=2) for u in range(J1+1)];
        
        return torch.cat(v,dim=1);
    
    def T_mat(self, screen, nframe, natm):

        Tmat = torch.mean(screen.reshape([nframe,natm,6]), axis=1);
        Tmat = torch.einsum('kl,il->ik',self.T_trans,Tmat);

        return Tmat.reshape([nframe,3,3])/natm;

    def raw_to_mat(self, V_raw, minibatch, labels):
        
        node_H = self.transform(V_raw['H'],1,1);
        node_C = self.transform(V_raw['C'],2,2);
        edge_HH = self.transform(V_raw['HH'],1,1);
        edge_CH = self.transform(V_raw['CH'],2,1);
        edge_CC = self.transform(V_raw['CC'],2,2);
        screen = V_raw['screen'];
        gap = V_raw['gap'];

        norbs = labels['norbs'];
        nframe = labels['nframe'];
        map1 = labels['map1'];
        num_nodes = minibatch['num_nodes'];
        batch = labels['batch'];
        natm = labels['natm'];
        f_in = minibatch['f_in'];
        HH_ind = minibatch['HH_ind'];
        CC_ind = minibatch['CC_ind'];
        CH_ind = minibatch['CH_ind'];
        edge_src = minibatch['edge_src'];
        edge_dst = minibatch['edge_dst'];
        
        Vmat = [torch.zeros([norbs, norbs], dtype=torch.float).to(self.device) for i in range(nframe)];
        gap_mat = [torch.zeros([natm, natm, 3], dtype=torch.float).to(self.device) for i in range(nframe)];
        Tmat = self.T_mat(screen, nframe, natm);
        
        for i in range(num_nodes):
            frame = batch[i];
            v = i-frame*natm;
            if(f_in[i,0]):
                Vmat[frame][map1[v]:map1[v+1],map1[v]:map1[v+1]] = node_C[i];
            else:
                Vmat[frame][map1[v]:map1[v+1],map1[v]:map1[v+1]] = node_H[i];

        for i in range(len(HH_ind)):
            u1,u2 = edge_src[HH_ind[i]],edge_dst[HH_ind[i]];
            frame = batch[u1];
            v1 = u1-frame*natm;
            v2 = u2-frame*natm;
            Vmat[frame][map1[v1]:map1[v1+1], map1[v2]:map1[v2+1]] = edge_HH[i];
            gap_mat[frame][v1,v2] = gap[1][i];

        for i in range(len(CC_ind)):
            u1,u2 = edge_src[CC_ind[i]],edge_dst[CC_ind[i]];
            frame = batch[u1];
            v1 = u1-frame*natm;
            v2 = u2-frame*natm;
            Vmat[frame][map1[v1]:map1[v1+1], map1[v2]:map1[v2+1]] = edge_CC[i];
            gap_mat[frame][v1,v2] = gap[0][i];

        for i in range(len(CH_ind)):
            u1,u2 = edge_src[CH_ind[i]],edge_dst[CH_ind[i]];
            frame = batch[u1];
            v1 = u1-frame*natm;
            v2 = u2-frame*natm;
            Vmat[frame][map1[v1]:map1[v1+1], map1[v2]:map1[v2+1]] = edge_CH[i];
            gap_mat[frame][v1,v2] = gap[2][i];
        
        Vmat = torch.stack([(V+V.T)/2 for V in Vmat]);
        gap = torch.stack([G for G in gap_mat]).reshape([nframe,natm**2,3]);

        gap = torch.sum(torch.nn.Softmax(dim=1)(gap[:,:,0:1])*gap[:,:,1:], axis=1);

        return Vmat, Tmat, gap;
        
        