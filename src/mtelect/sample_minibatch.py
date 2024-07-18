# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:31:52 2023

@author: haota
"""

import numpy as np;
import torch;
from torch_cluster import radius_graph;
from e3nn import o3;

class sampler(object):

    def __init__(self, data_in, labels, device, min_radius: float = 0.5, max_radius: float = 2):

        self.data = data_in;
        self.labels = labels;
        self.device = device;
        self.max_radius = max_radius;
        self.min_radius = min_radius;
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2);
        self.element_embedding = {'H':[0,1],'C':[1,0]};

    def sample(self, batch_size, i_molecule, op_names=[]):

        data = self.data[i_molecule];

        ind = [i for i in range(len(self.data[i_molecule]['pos']))];
#        np.random.shuffle(ind);
        ind = ind[:batch_size];
        natm = len(data['elements']);
        nframe = len(ind);
        num_nodes = nframe*natm;

        pos = data['pos'][ind].reshape([-1,3]);
        batch = torch.tensor([int(i//natm) for i in range(num_nodes)]).to(self.device);
        edge_src, edge_dst = radius_graph(x=pos, r=self.max_radius, batch=batch);
        self_edge = torch.tensor([i for i in range(num_nodes)]).to(self.device);
        edge_src = torch.cat((edge_src, self_edge));
        edge_dst = torch.cat((edge_dst, self_edge));
        edge_vec = pos[edge_src] - pos[edge_dst];
        num_neighbors = len(edge_src) / num_nodes;
        sh = o3.spherical_harmonics(l = self.irreps_sh, 
                                    x = edge_vec, 
                                    normalize=True, 
                                    normalization='component').to(self.device)

        rnorm = edge_vec.norm(dim=1);
        crit1, crit2 = rnorm<self.max_radius, rnorm>self.min_radius;
        emb = (torch.cos(rnorm/self.max_radius*torch.pi)+1)/2; 
        emb = (emb*crit1*crit2 + (~crit2)).reshape(len(edge_src),1);

        f_in = torch.tensor([self.element_embedding[u] for u in data['elements']]*nframe,
                            dtype=torch.float).to(self.device);
        CC_ind = torch.argwhere(f_in[edge_src][:,0]*f_in[edge_dst][:,0]).reshape(-1);
        HH_ind = torch.argwhere(f_in[edge_src][:,1]*f_in[edge_dst][:,1]).reshape(-1);
        CH_ind = torch.argwhere(f_in[edge_src][:,0]*f_in[edge_dst][:,1]).reshape(-1);

        map1 = [5+9*(ele=='C') for ele in data['elements']];
        map1 = [sum(map1[:i]) for i in range(len(map1)+1)];

        minibatch = {
                     'sh': sh,
                     'emb': emb,
                     'f_in':f_in,
                     'edge_src': edge_src,
                     'edge_dst': edge_dst,
                     'num_nodes': num_nodes,
                     'num_neighbors':num_neighbors,
                     'HH_ind': HH_ind,
                     'CH_ind': CH_ind,
                     'CC_ind': CC_ind
                     };

        batch_labels = {
            'norbs': data['properties']['norbs'],
            'nframe': len(ind),
            'batch': batch,
            'map1': map1,
            'natm': natm,
            'h': self.labels[i_molecule]['h'][ind],
            'Ee': self.labels[i_molecule]['E'][ind]-self.labels[i_molecule]['E_nn'][ind],
            'ne': data['properties']['ne'],
            'atomic_charge': self.labels[i_molecule]['atomic_charge'][ind],
            'E_gap': self.labels[i_molecule]['E_gap'][ind],
            'B': self.labels[i_molecule]['B'][ind],
            'alpha': self.labels[i_molecule]['alpha'][ind]
            };

        for op_name in op_names:
            batch_labels[op_name] = self.labels[i_molecule][op_name][ind]

        return minibatch, batch_labels;
