import os;
import numpy as np;
import json;
from ase.io.cube import read_cube_data;
import scipy;

Nframe = 500;

Elist = [];
posl  = [];
atm   = [];
nr = [];
grid = [];

E_orb = [];
E_nn = [];
E_HF = [];

route = os.getcwd()+'/';
u = -2;
while(route[u]!='/'):
    u -= 1;
name = route[u+1:-1];

res = os.popen("grep 'Number of Electrons' pvdz/0/log").readline();
ne = float(res.split()[-1]);
print('N electrons:' + str(ne));

os.chdir(route+'pvtz/');
for i in range(Nframe):
    print('reading basic information: '+str(i));
    res = os.popen("grep 'E(CCSD(T))' "+str(i)+'/log').readline();
    if(len(res)!=0):
        E1 = float(res.split()[-1][:-1]);
    else:
        E1 = False;
    Elist.append(E1);

    res = os.popen("grep 'SINGLE POINT ENERGY' ../pvdz/"+str(i)+'/log').readline();
    if(len(res)!=0):
        E1 = float(res.split()[-1][:-1]);
    else:
        E1 = False;
    E_HF.append(E1);

    res = os.popen("grep 'One Electron Energy' ../pvdz/"+str(i)+'/log').readline();
    E_orb.append(float(res.split()[-4]));
    res = os.popen("grep 'Nuclear Repulsion' ../pvdz/"+str(i)+'/log').readline();
    E_nn.append(float(res.split()[-2]));

    posl.append([]);
    atm.append([]);
    with open(str(i)+'/run.inp','r') as file:
        data = file.readlines();
        n,test = 0,'';
        while(test != '* xyz'):
            n += 1;
            test = data[-n-2][:5];
        data = data[-1-n:-1];
        for dp in data:
            posl[-1].append([float(u[:-1]) for u in dp.split()[1:]]);
            atm[-1].append(dp.split()[0]); 
                       
    if('run.eldens.cube' in os.listdir(str(i))):
        data, atoms = read_cube_data(str(i)+'/run.eldens.cube');
        data = data[20:80,20:80,20:80].tolist();
        with open(str(i)+'/run.eldens.cube','r') as file:
            res = file.readlines();
            init = [float(x) for x in res[2].split()[1:]];
            dist = [float(res[3].split()[1]), float(res[4].split()[2]), float(res[5].split()[3])];
            init = [init[i]+dist[i]*20 for i in range(3)];
            grid.append([init,dist]);
            nr.append(data)
    else:
        nr.append(False);
        grid.append(False);

def readmat(data):
    number = int(data[-1].split()[0])+1;
    rep = int(round(len(data)/(number+1)));

    matl = [];
    for i in range(rep):
        res = [[float(t) for t in s.split()[1:]] for s in data[i*(number+1)+1:i*(number+1)+number+1]];
        matl.append(np.array(res));
        
    mat = np.hstack(matl)
    return mat;

mlist = [];
slist = [];
os.chdir(route);

for i in range(Nframe):
    print('reading matrix information: '+str(i));
    with open('pvdz/' + str(i)+'/log', 'r') as file:
        output =  file.readlines();
        u =  0;
        try:
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

            h = readmat(dataf);
            S = readmat(dataS);

            h += (-np.sum(scipy.linalg.eigvalsh(h,S)[:int(ne/2)])*2 + E_HF[i] - E_nn[i])/ne*S;

            slist.append(S.tolist());
            mlist.append(h.tolist());
        except:
            raise
            slist.append(False);
            mlist.append(False);
            print('convergence failure '+str(i));

output = {'coordinates':posl, 'HF': E_HF, 'elements':atm, 'S':slist, 'h':mlist, 'energy':Elist, 'Enn':E_nn};

with open(name+'_data.json','w') as file:
    json.dump(output, file);


pos = -2;
while(route[pos]!='/'):
    pos -= 1;
name = route[pos+1:-1];
try:
    with open(route+name+'_data.json','r') as file:
        data = json.load(file);
except:
    data = {};

OPS = ['x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'yz', 'xz'];
obs_dic = {key:[] for key in OPS};
obs_dic['atomic_charge'] = [];
obs_dic['bond_order'] = [];
obs_dic['Ee'] = [];
obs_dic['T']  = [];
obs_dic['alpha'] = [];
natom = len(data['elements'][0]);

for i in range(500):
    
    print('reading observables: '+str(i));
    try:
        dipole = os.popen("grep -A 4 'Electronic Contribution' "+route+'pvtz/'+str(i)+'/run_property.txt');
        dipole = [-float(u[:-1].split()[-1]) for u in dipole.readlines()[-3:]];
        
        obs_dic['x'].append(dipole[0]);
        obs_dic['y'].append(dipole[1]);
        obs_dic['z'].append(dipole[2]);
        
        quadrupole = os.popen("grep -A 4 'Electronic part' "+route+'pvtz/'+str(i)+'/run_property.txt');
        quadrupole = [[-float(v) for v in u[:-1].split()[1:]] for u in quadrupole.readlines()[-3:]];
        
        obs_dic['xx'].append(quadrupole[0][0]);
        obs_dic['yy'].append(quadrupole[1][1]);
        obs_dic['zz'].append(quadrupole[2][2]);
        obs_dic['xy'].append(quadrupole[0][1]);
        obs_dic['yz'].append(quadrupole[1][2]);
        obs_dic['xz'].append(quadrupole[0][2]);
        
        command = "grep -A "+str(natom+1)+" 'MULLIKEN ATOMIC CHARGE' ";
        atomicCharge = os.popen(command +route+'pvtz/'+str(i)+'/log').readlines()[-natom:];
        atomicCharge = [float(u[:-1].split()[-1]) for u in atomicCharge];
        obs_dic['atomic_charge'].append(atomicCharge);
        
        command = "grep -A "+str(20)+" 'Mayer bond orders' ";
        bond_data = os.popen(command +route+'pvtz/'+str(i)+'/log').readlines();
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
            
        obs_dic['bond_order'].append(bond_order);
        
        dp = os.popen("grep -A 7 'ABSORPTION SPECTRUM' "+route+'/eom/'+str(i)+'/log').readlines();
        dp = [[float(v) for v in u.split()] for u in dp[-3:]];
        obs_dic['Ee'].append([u[1] for u in dp]);
        obs_dic['T'].append([[u[5],u[6],u[7]] for u in dp]);
        
        dp = os.popen("grep -A 3 'The raw cartesian tensor' "+route+"polar/"+str(i)+"/log").readlines();
        dp = [[float(v) for v in u.split()] for u in dp[-3:]];
        obs_dic['alpha'].append(dp);

    except:
        
        for key in obs_dic:
            obs_dic[key].append(False);

for key in obs_dic:
    data[key] = obs_dic[key];

for i in range(len(data['energy'])):
    j = len(data['energy']) - i - 1;
    if(data['S'][j]==False):
        for key in data:
            del data[key][j];
        print('removing convergence failure: '+str(j));
    
    elif(data['energy'][j]==False):
        for key in data:
            del data[key][j];
        print('removing convergence failure: '+str(j));

with open(route+name+'_obs_data.json','w') as file:
    json.dump(data, file);

