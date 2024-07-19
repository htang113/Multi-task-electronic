import numpy as np;
import os;
import json;
import scipy;

class QM_reader(object):

    def __init__(self, route):

        self.route = route;

    def read_ne(self, folder):

        path  = self.route + folder + '/log';
        res = os.popen("grep 'Number of Electrons' " +path).readline();
        ne = float(res.split()[-1]);
        print('N electrons:' + str(ne));
        self.ne = ne;
        return ne;

    def read_HF(self, folder):

        path  = self.route + folder + '/log';
        print('reading basic information: '+folder);

        res = os.popen("grep 'SINGLE POINT ENERGY' " + path).readline();
        if(len(res)!=0):
            E1 = float(res.split()[-1][:-1]);
        else:
            E1 = False;
        self.E_HF = E1;

        res = os.popen("grep 'Nuclear Repulsion' " + path).readline();
        self.E_nn = float(res.split()[-2]);

        self.posl = [];
        self.atm = [];
        with open(self.route + folder+'/run.inp','r') as file:
            data = file.readlines();
            n,test = 0,'';
            while(test != '* xyz'):
                n += 1;
                test = data[-n-2][:5];
            data = data[-1-n:-1];
            for dp in data:
                self.posl.append([float(u[:-1]) for u in dp.split()[1:]]);
                self.atm.append(dp.split()[0]); 
        self.natom = len(self.posl);
        
        return {'HF': self.E_HF, 'coordinates': self.posl, 'elements': self.atm, 'Enn': self.E_nn};
    
    def readmat(self, data):
        number = int(data[-1].split()[0])+1;
        rep = int(round(len(data)/(number+1)));

        matl = [];
        for i in range(rep):
            res = [[float(t) for t in s.split()[1:]] for s in data[i*(number+1)+1:i*(number+1)+number+1]];
            matl.append(np.array(res));
            
        mat = np.hstack(matl)
        return mat;

    def read_matrix(self, folder):
        
        path  = self.route + folder + '/log';
        print('reading matrix information: '+folder);
        with open(path, 'r') as file:
            output =  file.readlines();
            u =  0;

            while('signals convergence' not in output[u]):
                u += 1;
                if('OVERLAP MATRIX' in output[u]):
                    s = int(u)+1;
                if('Time for model grid setup' in output[u]):
                    t = int(u);
            v = int(u);
            while('Fock matrix for operator 0' not in output[v]):
                v -= 1;
            dataf = output[v+1:u];
            dataS = output[s+1:t];

            h = self.readmat(dataf);
            S = self.readmat(dataS);

            h += (-np.sum(scipy.linalg.eigvalsh(h,S)[:int(self.ne/2)])*2 + self.E_HF - self.E_nn)/self.ne*S;

        return {'S': S.tolist(), 'h': h.tolist()};
    
    def read_obs(self, folder):
        
        path  = self.route + folder + '/';
        obs_dic = {};
        dipole = os.popen("grep -A 4 'Electronic Contribution' " + path + 'run_property.txt');
        dipole = [-float(u[:-1].split()[-1]) for u in dipole.readlines()[-3:]];
        
        obs_dic['x'] = dipole[0];
        obs_dic['y'] = dipole[1];
        obs_dic['z'] = dipole[2];
        
        quadrupole = os.popen("grep -A 4 'Electronic part' " + path + 'run_property.txt');
        quadrupole = [[-float(v) for v in u[:-1].split()[1:]] for u in quadrupole.readlines()[-3:]];
        
        obs_dic['xx'] = quadrupole[0][0];
        obs_dic['yy'] = quadrupole[1][1];
        obs_dic['zz'] = quadrupole[2][2];
        obs_dic['xy'] = quadrupole[0][1];
        obs_dic['yz'] = quadrupole[1][2];
        obs_dic['xz'] = quadrupole[0][2];
        
        command = "grep -A "+str(self.natom+1)+" 'MULLIKEN ATOMIC CHARGE' ";
        atomicCharge = os.popen(command + path + 'log').readlines()[-self.natom:];
        atomicCharge = [float(u[:-1].split()[-1]) for u in atomicCharge];
        obs_dic['atomic_charge'] = atomicCharge;
        
        command = "grep -A "+str(20)+" 'Mayer bond orders' ";
        bond_data = os.popen(command + path + 'log').readlines();
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
        
        dp = os.popen("grep -A 7 'ABSORPTION SPECTRUM' "+ path +'log').readlines();
        dp = [[float(v) for v in u.split()] for u in dp[-3:]];
        obs_dic['Ee'] = [u[1] for u in dp];
        obs_dic['T'] = [[u[5],u[6],u[7]] for u in dp];
        
        dp = os.popen("grep -A 3 'The raw cartesian tensor' "+ path +"log").readlines();
        dp = [[float(v) for v in u.split()] for u in dp[-3:]];
        obs_dic['alpha'] = dp;
        
        return obs_dic;
