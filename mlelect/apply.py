from mtelect.integral import integrate
from mtelect.model import V_theta
from mtelect.predictor import predict_fns
import numpy as np
import torch
from mtelect.tomat import to_mat;
import os;
import json;
import matplotlib;
import matplotlib.pyplot as plt;
import scipy;
from mtelect.integral import integrate;
from torch_cluster import radius_graph;
from e3nn import o3;

class estimator_apply():

    def __init__(self, device, output_folder='test_output', scaling = 0.1) -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.integrator = integrate(device)
        self.loss = torch.nn.MSELoss();
        self.transformer = to_mat(device);
        self.output_folder = output_folder;
        self.scaling = scaling;

        if(not os.path.exists(output_folder)):
            os.mkdir(output_folder);
    
    def load(self, filename):

        self.model = V_theta(self.device).to(self.device)

        try:
            self.model.load_state_dict(torch.load(filename, map_location=torch.device(self.device)));
        except:
            res = torch.load(filename, map_location=torch.device(self.device));
            for key in list(res.keys()):
                res[key[7:]] = res[key];
                del res[key];
            self.model.load_state_dict(res);

    def set_op_matrices(self, op_matrices):
        self.op_matrices = op_matrices

    def solve(self, minibatch, labels, obs_mat, E_nn, data_in,
              op_names=None, Cmat = [], save_filename='data') -> float:

        angstron2Bohr = 1.88973
        h = labels['h'];

        # number of occupied orbitals
        ne = labels['ne'];
        nbasis = labels['norbs']
        nframe = labels['nframe']
        V_raw = self.model(minibatch);

        V, T, G = self.transformer.raw_to_mat(V_raw,minibatch,labels);
        V *= self.scaling['V'];
        T *= self.scaling['T'];

        pred = predict_fns(h, V, ne, nbasis, nframe, self.device);

        with open(self.output_folder + save_filename + '_H.json','w') as file:
            json.dump(pred.H.tolist(), file)

        elements = data_in['elements'];
        nuclearCharge = [1+5*(ele=='C') for ele in elements];
        nuclearCharge = torch.tensor(nuclearCharge, dtype=torch.float).to(self.device);
        mass = torch.tensor([1.008 + (12.011-1.008)*(ele=='C') for ele in elements], dtype=torch.float).to(self.device);
        pos = data_in['pos'];
        mass_center = torch.sum(pos*mass[None,:,None], axis=1)/torch.sum(mass);
        pos = (pos - mass_center[:,None,:])*angstron2Bohr;

        obj = {};
        obj['C'] = elements.count('C');
        obj['H'] = elements.count('H');
        
        for i, op_name in enumerate(op_names):

            if(op_name == 'E'):

                Ehat = pred.E(E_nn);
                obj[op_name] = {'Ehat':Ehat.tolist()};

            if(op_name in ['x','y','z','xx','yy','zz','xy','xz','yz']):

                moment = torch.tensor([op_name.count('x'),
                            op_name.count('y'),
                            op_name.count('z')], dtype=torch.float).to(self.device);
                multipole = torch.sum(torch.prod(pos**moment[None,None,:],
                                                 axis=2)*nuclearCharge[None,:], axis=1);
                
                O_mat = obs_mat[op_name]
                Ohat = pred.O(O_mat)
                Ohat = multipole - Ohat;
                obj[op_name] = {'Ohat': Ohat.tolist()};
        
            if(op_name == 'atomic_charge'):

                Chat = pred.C(Cmat);
                Chat = nuclearCharge[None,:]-Chat;
                obj['atomic_charge'] = {'Chat': Chat.tolist()};

            if(op_name == 'E_gap'):

                Eg_hat = pred.Eg(G);
                obj['E_gap'] = {'Eg_hat': Eg_hat.tolist()};

            if(op_name == 'bond_order'):
                    
                Bhat = pred.B(Cmat);
                obj['bond_order'] = {'Bhat': Bhat.tolist()};
            
            if(op_name == 'alpha'):

                r_mats = torch.stack([obs_mat['x'],
                                      obs_mat['y'],
                                      obs_mat['z']]);
                alpha_hat = pred.alpha(r_mats, T, G);
                obj['alpha'] = {'alpha_hat': alpha_hat.tolist()};

        with open(self.output_folder+'/'+save_filename+'.json','w') as file:
            json.dump(obj,file)

        return Ehat;


def load_data_apply(data, device, load_obs_mat = True):

    data_in = [];
    labels = [];
    obs_mats = [];
    op_names = ['x','y','z','xx','yy', 'zz','xy','xz','yz']
    molecules = data['name'];
    for i,molecule in enumerate(molecules):
        
        pos = torch.tensor(data['coordinates'][i]).to(device);
        pos[:,[0,1,2]] = pos[:,[1,2,0]];
        elements = data['elements'][i];

        ne = int(round(sum([1+5*(ele=='C') for ele in elements])/2));
        norbs = int(round(sum([5+9*(ele=='C') for ele in elements])));
        nframe = 1;

        data_in.append({'pos':pos[None,:,:],'elements':elements,
                    'properties':{'ne':ne, 'norbs':norbs,'nframe':nframe}});

        h = torch.tensor(data['h'][i]).to(device);
        S_mhalf = scipy.linalg.fractional_matrix_power(data['S'][i], (-1/2)).tolist();
        S_mhalf = torch.tensor([S_mhalf]).to(device);
        h = torch.matmul(torch.matmul(S_mhalf, h),S_mhalf);
        
        label = {'S': torch.tensor([data['S'][i]]).to(device),
                 'Smhalf': S_mhalf,
                 'h': h,
                 'E_nn':torch.tensor([data['Enn'][i]]).to(device)};
        labels.append(label);
        integrator = integrate(device);

        data_obs = {}
        for operator in op_names:

            res = [];
            res += integrator.calc_O(pos[None,:,:],elements,operator);
            data_obs[operator] = res;

        obs_mats.append({op: torch.matmul(torch.matmul(S_mhalf,torch.Tensor(data_obs[op]).to(device)),S_mhalf)
                                                        for op in op_names})

    return data_in, labels, obs_mats;


class sampler_apply(object):

    def __init__(self, data_in, labels, device, min_radius: float = 0.5, max_radius: float = 2):

        self.data = data_in;
        self.labels = labels;
        self.device = device;
        self.max_radius = max_radius;
        self.min_radius = min_radius;
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2);
        self.element_embedding = {'H':[0,1],'C':[1,0]};

    def sample(self, batch_size, i_molecule):

        data = self.data[i_molecule];
        op_names = ['x','y','z','xx','yy', 'zz','xy','xz','yz'];
        natm = len(data['elements']);
        nframe = 1;
        num_nodes = nframe*natm;

        pos = data['pos'].reshape([-1,3]);
        batch = torch.tensor([0 for i in range(num_nodes)]).to(self.device);
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

        f_in = torch.tensor([self.element_embedding[u] for u in data['elements']],
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
            'nframe': 1,
            'batch': batch,
            'map1': map1,
            'natm': natm,
            'h': self.labels[i_molecule]['h'],
            'ne': data['properties']['ne']
            };

        return minibatch, batch_labels;
