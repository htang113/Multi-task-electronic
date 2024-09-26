import json;
import torch;
import scipy;
import os;
import numpy as np;

from basis.Irreps import Irreps_build;
from basis.integral import integrate;

class dataloader():

    def __init__(self, device, element_list, 
                 path, batch_size = 100, 
                 starting_basis = 'def2-SVP'):
        
        self.device = device;
        self.path = path +'/';
        self.batch_size = batch_size;
        irreps = Irreps_build(element_list);
        self.elements = element_list;
        self.nbasis = irreps.get_nbasis();
        self.ne_dic = irreps.get_ne();
        self.op_names = ['x','y','z','xx','yy','zz','xy','xz','yz'];

        self.integrator = integrate(device, starting_basis=starting_basis, path=path[:-7]);

    def read_basic(self, filepath):

        with open(filepath, 'r') as f:
            basic = json.load(f);
        pos = torch.tensor(basic['coordinates']).to(self.device);
        pos = pos[:,[1,2,0]];
        elements = basic['elements'];

        self.pos = pos;
        self.elements = elements;
        nuclearCharge = [self.ne_dic[ele] for ele in elements];
        self.nuclearCharge = torch.tensor(nuclearCharge, dtype=torch.float).to(self.device);

        ne = int(round(sum([self.ne_dic[ele] for ele in elements])/2));
        norbs = int(round(sum([self.nbasis[ele] for ele in elements])));

        h = torch.tensor(basic['h']).to(self.device);
        S_mhalf = scipy.linalg.fractional_matrix_power(basic['S'], (-1/2)).tolist();
        S_mhalf = torch.tensor(S_mhalf).to(self.device);
        self.S_mhalf = S_mhalf;
        h = torch.matmul(torch.matmul(S_mhalf, h),S_mhalf);

        return {'pos':pos,'elements':elements, 'S': torch.tensor(basic['S']).to(self.device),
                'Smhalf': S_mhalf, 'h': h, 'E_nn':basic['Enn'],
                'ne':ne, 'norbs':norbs};

    def read_obs(self, filepath):

        with open(filepath, 'r') as f:
            obs = json.load(f);

        label = {};

        if('energy' in obs):
            label['E'] = obs['energy'];
            label['E_nn'] = obs['E_nn'];
        if('Ee' in obs):
            cm_inverse_to_hartree = 4.55633528*1E-6;
            label['E_gap'] = cm_inverse_to_hartree * obs['Ee'][0];

        if('atomic_charge' in obs):
            label['atomic_charge'] = - torch.tensor(obs['atomic_charge']).to(self.device) + self.nuclearCharge;

        if('bond_order' in obs):
            B_labels = torch.zeros([len(self.elements), 
                                    len(self.elements)]).to(self.device);
            for dp in obs['bond_order']:
                B_labels[dp[0], dp[1]] = dp[2];
                B_labels[dp[1], dp[0]] = dp[2];
            label['B'] = B_labels;

        if('alpha' in obs):
            label['alpha'] = torch.tensor(obs['alpha']).to(self.device);
        
        for op_name in self.op_names:
            if(op_name in obs):
                label[op_name] = obs[op_name]

        return label;

    def read_obs_mat(self):

        data_obs = {}
        for operator in self.op_names:
            res = self.integrator.calc_O(self.pos[None,:,:], 
                                         self.elements, operator)[0];
            output = torch.matmul(torch.matmul(self.S_mhalf,
                                               torch.Tensor(res).to(self.device)),
                                  self.S_mhalf)
            data_obs[operator] = output;

        return data_obs;

    def get_files_old(self, path, rank, world_size):

        fl = os.listdir(path);
        
        if(world_size == 1):

            return fl, self.batch_size;
        
        else:

            Nbasis = np.array([self.nbasis[el] for el in self.elements]);
            fmat = [];

            for f in fl:

                res = f.split('_')[1:];
                res[-1] = res[-1][:-5];
                fmat.append([int(r) for r in res]);

            Nbl = np.sum(np.array(fmat) * Nbasis[None,:], axis=1);

            indices = np.argsort(Nbl);
            Nb_sorted = Nbl[indices];
            load = np.sum(Nbl)/world_size;

            start = rank*load;
            end = (rank+1)*load;

            for i in range(len(Nb_sorted)):
                
                numbers = np.sum(Nb_sorted[:i])
                if(numbers >= start and i <= start):
                    start = i;
                if(numbers >= end or i==len(Nb_sorted)-1):
                    end = i;
                    break;

            files = [fl[i] for i in indices[start:end]];
            partition = int(self.batch_size * (end-start) / len(fl));

            print('Rank: ', rank, 'start: ', start, 'end: ', end, 'partition: ', partition)
            return files, partition;
    
    def get_files(self, path_list, rank, world_size):

        fl = [];
        for path in path_list:
            fl += [(path, f) for f in os.listdir(path)];
        
        if(world_size == 1):

            return fl, self.batch_size;
        
        else:

            files = fl[len(fl)*rank//world_size: len(fl)*(rank+1)//world_size]
            partition = int(self.batch_size/world_size);

            return files, partition;

    def load_data(self, group, rank, world_size):
        
        path_list = [self.path + g + '/basic/' for g in group];
        fl, partition = self.get_files(path_list, rank, world_size);
        
        data_in = [];
        labels = [];
        obs_mat = [];

        for file in fl:
            
            basic_path = file[0] + file[1];
            data_in.append(self.read_basic(basic_path));
            obs_path = file[0][:-7] + '/obs/' + file[1].split('_')[0]+file[1][-5:];
            labels.append(self.read_obs(obs_path));
            obs_mat.append(self.read_obs_mat());

        return data_in, labels, obs_mat, partition;
            