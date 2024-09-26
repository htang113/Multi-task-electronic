from basis.integral import integrate
from model.model_cls import V_theta
from deploy.predictor import predict_fns
from model.sample_minibatch import sampler as sample_train
from data.loader import dataloader
import numpy as np
import torch
from model.tomat import to_mat;
import os;
import json;
import matplotlib;
import matplotlib.pyplot as plt;
import scipy;
from torch_cluster import radius_graph;
from e3nn import o3;
from deploy.deploy_cls import estimator_test
from periodictable import elements

class estimator(estimator_test):

    def __init__(self, device, data_in, op_matrices, 
                 output_folder='output/inference') -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.sampler = sampler(data_in, device);
        self.n_molecules = len(data_in);
        self.op_matrices = op_matrices;
        self.el_dict = {el.symbol: {'Z':el.number,'M':el.mass} for el in elements};

        if(not os.path.exists(output_folder)):
            os.mkdir(output_folder);

        return None;

    def get_nuclear_charge(self, batch_ind = 0, batch_size=1):

        ind = list(range(len(self.sampler.data)));
        ind = ind[batch_ind*batch_size : (batch_ind+1)*batch_size];
        data = [self.sampler.data[i] for i in ind];
        self.nuclearCharge = [];

        for dp in data:

            elements = dp['elements'];
            nuclearCharge = [self.el_dict[ele]['Z'] for ele in elements];
            self.nuclearCharge.append(torch.tensor(nuclearCharge, 
                                      dtype=torch.float).to(self.device));

        return self.nuclearCharge;

    def get_centered_pos(self, batch_ind = 0, batch_size=1):

        ind = list(range(len(self.sampler.data)));
        ind = ind[batch_ind*batch_size : (batch_ind+1)*batch_size];
        data = [self.sampler.data[i] for i in ind];
        self.pos = [];

        angstron2Bohr = 1.88973;
        for dp in data:
            pos = dp['pos'];
            elements = dp['elements'];
            mass = torch.tensor([self.el_dict[ele]['M'] for ele in elements], 
                                dtype=torch.float).to(self.device);
            mass_center = torch.sum(pos*mass[:,None], axis=0)/torch.sum(mass);
            pos = (pos - mass_center[None,:])*angstron2Bohr;
            self.pos.append(pos);

        return self.pos;
    
    def get_multipole(self, op_name, batch_ind = 0, batch_size=1):

        pos = self.get_centered_pos(batch_ind = batch_ind, batch_size=batch_size);
        nuclearCharge = self.get_nuclear_charge(batch_ind = batch_ind, batch_size=batch_size);
        moment = torch.tensor([op_name.count('x'),
                               op_name.count('y'),
                               op_name.count('z')], dtype=torch.float).to(self.device);

        self.multipole = [];
        for ind, posi in enumerate(pos):
            multipole = torch.sum(torch.prod(posi**moment[None,:],
                                                axis=1)*nuclearCharge[ind]);

            self.multipole.append(multipole);

        return self.multipole;


    def solve_apply(self, batch_size=1, op_names=[]) -> dict:
        
        operators_electric = [key for key in list(op_names.keys()) \
                if key in ['x','y','z','xx','yy',
                            'zz','xy','xz','yz']];
        N_batchs = int(round(self.n_molecules/batch_size));
        properties_dic = {};

        for batch_ind in range(N_batchs):

            minibatch, ind = self.sampler.sample(batch_ind = batch_ind, batch_size=batch_size,
                                                    irreps=self.irreps, op_names=operators_electric);
            
            V, T, G = self.inference(minibatch);
            
            # number of occupied orbitals
            h = minibatch['h'];
            ne = minibatch['ne'];
            norbs = minibatch['norbs'];
            property_calc = predict_fns(h, V, T, G, ne, norbs, self.device);
            
            properties = self.evaluate_properties(property_calc, ind, op_names);

            for i, op_name in enumerate(op_names):

                if(op_name in ['x','y','z','xx','yy','zz','xy','xz','yz']):

                    multipole = self.get_multipole(op_name, batch_ind = batch_ind, batch_size=batch_size);
                    Ohat = [(multipole[ind] - torch.tensor(ele_part, 
                            dtype=torch.float).to(self.device)).tolist() for ind,ele_part in enumerate(properties[op_name])];
                    properties[op_name] = Ohat;
            
                if(op_name == 'atomic_charge'):

                    nuclearCharge = self.get_nuclear_charge(batch_ind = batch_ind, batch_size=batch_size);

                    Chat = [(nuclearCharge[ind] - torch.tensor(ele_part, 
                            dtype=torch.float).to(self.device)).tolist() for ind,ele_part in enumerate(properties[op_name])];
                    properties[op_name] = Chat;
                
                if(op_name == 'E_gap'):
                    
                    hartree_to_eV = 27.211396641308;
                    properties[op_name] = [Ei*hartree_to_eV for Ei in properties[op_name]];
                
                if(op_name == 'E'):
                        
                    hartree_to_kcalmol = 627.509;
                    res = [];
                    for i1, Ei in enumerate(properties[op_name]):
                        res.append((Ei+self.sampler.data[ind[i1]]['E_nn'])*hartree_to_kcalmol);
                    properties[op_name] = res;
                
                if(op_name == 'bond_order'):

                    update = [];
                    for Bi in properties[op_name]:

                        Bind = np.argwhere(np.abs(np.array(Bi))>0.1);
                        Bnew = [(int(ind[0]), int(ind[1]), Bi[ind[0]][ind[1]]) for ind in Bind];
                        update.append(Bnew);
                    properties[op_name] = update;
                
                if(op_name not in properties_dic.keys()):
                    properties_dic[op_name] = properties[op_name];
                    properties_dic['name'] = [self.sampler.data[i]['name'] for i in ind];
                else:
                    properties_dic[op_name] += properties[op_name];
                    properties_dic['name'] += [str(self.sampler.data[i]['name']) for i in ind];

        return properties_dic;

class load_data(dataloader):

    def __init__(self, device, element_list, 
                 path, starting_basis = 'def2-SVP') -> None:

        dataloader.__init__(self, device=device, element_list = element_list,
                            path = path, batch_size = None, 
                            starting_basis = starting_basis);

    def load(self, group):

        path_list = [self.path + g + '/basic/' for g in group];
        fl, partition = self.get_files(path_list, 0, 1);
        
        data_in = [];
        obs_mat = [];

        for file in fl:
            
            basic_path = file[0] + file[1];
            data_in.append(self.read_basic(basic_path));
            obs_mat.append(self.read_obs_mat());
            data_in[-1]['name'] = file[1];

        return data_in, obs_mat;



class sampler(sample_train):

    def __init__(self, data_in, device) -> None:

        sample_train.__init__(self, data_in, [], device);
    
    def sample(self, batch_ind, batch_size, irreps, op_names=[]):

        ind = list(range(len(self.data)));
        ind = ind[batch_ind*batch_size : (batch_ind+1)*batch_size];
        data = [self.data[i] for i in ind];

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

        return minibatch, ind;