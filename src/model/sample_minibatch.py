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

    def count(self, batch, tensor):

        Nedge = torch.sum(tensor[None, :] == batch[:, None], dim=1)
        Nnodes = torch.sum(batch[None, :] == batch[:, None], dim=1)

        return Nedge/Nnodes;

    def radius_embedding(self, rnorm):

        crit1, crit2 = rnorm<self.max_radius, rnorm>self.min_radius;
        emb = (torch.cos(rnorm/self.max_radius*torch.pi)+1)/2; 
        emb = (emb*crit1*crit2 + (~crit2)).reshape(len(emb),1);

        return emb
    
    def element_embedding(self, elements, irreps):

        element_embed = irreps.get_onehot();
        f_in = [];
        for ele in elements:
            f_in += [element_embed[u] for u in ele];
        f_in = torch.tensor(f_in, dtype=torch.float).to(self.device);

        return f_in;

    def get_map(self, elements, irreps):

        nbasis = irreps.get_nbasis();
        map1 = [];
        for ele in elements:
            mapi = [nbasis[e1] for e1 in ele];
            mapi = [sum(mapi[:i]) for i in range(len(mapi)+1)];
            map1.append(mapi);

        return map1;

    def get_list(self, labels, op_name):
        if(op_name in ['B','atomic_charge']):
            return [l[op_name] for l in labels];

        return [l[op_name] for l in labels];

    def get_tensor(self, data, op_name, basis_max):
        
        tensor = torch.zeros([len(data), basis_max, basis_max]);
        for i in range(len(data)):
            tensor[i, :data[i]['norbs'], :data[i]['norbs']] = data[i][op_name];

        return tensor.to(self.device);

    def sample(self, batch_ind, batch_size, irreps, op_names=[]):

        ind = list(range(len(self.data)));
#        np.random.shuffle(ind);
        ind = ind[batch_ind*batch_size : (batch_ind+1)*batch_size];
        data = [self.data[i] for i in ind];
        labels = [self.labels[i] for i in ind];

        elements = [u['elements'] for u in data];
        natms = [len(u) for u in elements];
        num_nodes = sum(natms);

        pos = torch.vstack([dp['pos'] for dp in data]);
        batch = torch.cat([torch.tensor([i]*n).to(self.device) for i,n in enumerate(natms)]);
        edge_src, edge_dst = radius_graph(x=pos, r=self.max_radius, batch=batch);
        self_edge = torch.tensor([i for i in range(num_nodes)]).to(self.device);
        edge_src = torch.cat((edge_src, self_edge));
        edge_dst = torch.cat((edge_dst, self_edge));
        edge_vec = pos[edge_src] - pos[edge_dst];

        num_neighbors = self.count(batch, batch[edge_src]);

        sh = o3.spherical_harmonics(l = self.irreps_sh, 
                                    x = edge_vec, 
                                    normalize=True, 
                                    normalization='component').to(self.device)

        rnorm = edge_vec.norm(dim=1);
        emb = self.radius_embedding(rnorm);
        f_in = self.element_embedding(elements, irreps);

        pair_ind = [];
        nele = irreps.get_input_irreps();
        for i in range(nele):
            for j in range(i+1):
                ind1 = torch.argwhere(f_in[edge_src][:,i]*f_in[edge_dst][:,j]).reshape(-1);
                pair_ind.append(ind1);

        map1 = self.get_map(elements, irreps);
        basis_max = max([dp['norbs'] for dp in data]);

        minibatch = {
                     'sh': sh,
                     'emb': emb,
                     'f_in':f_in,
                     'edge_src': edge_src,
                     'edge_dst': edge_dst,
                     'num_nodes': num_nodes,
                     'num_neighbors':num_neighbors,
                     'pair_ind': pair_ind,
                     'norbs': [dp['norbs'] for dp in data],
                     'batch': batch,
                     'map1': map1,
                     'natm': natms,
                     'ne': [dp['ne'] for dp in data],
                     'h': self.get_tensor(data, 'h', basis_max)
                     };

        batch_labels = {};
        if('E' in labels[0]):
            batch_labels['Ee'] = [l['E']-l['E_nn'] for l in labels]
        
        op_names += ['atomic_charge','E_gap','B','alpha'];

        for op_name in op_names:
            batch_labels[op_name] = self.get_list(labels, op_name);

        return minibatch, batch_labels, ind;
