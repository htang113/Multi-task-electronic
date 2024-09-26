from periodictable import elements
from itertools import product;
from e3nn import o3;

class Irreps_build(object):

    def __init__(self, elements_list = ['C','H']):
        
        self.elements = elements_list;
        self.el_dict = {el.symbol: el.number for el in elements};
        self.orbs = {};
    
        for el in self.elements:
            Z = self.el_dict[el];
            if(Z <=2):
                orb = [2,1,0,0];
            elif(Z <=4):
                orb = [3,2,0,0];
            elif(Z <=10):
                orb = [3,2,1,0];
            elif(Z <=11):
                orb = [4,2,1,0];
            elif(Z <=20):
                orb = [4,3,1,0];
            elif(Z <=22):
                orb = [5,3,2,0];
            elif(Z <=30):
                orb = [5,4,2,1];
            elif(Z<=36):
                orb = [5,4,3,0];
            else:
                raise "Element " + el + " not supported";
            self.orbs[el] = orb;

    def get_orbs(self):
        
        orbs = [];
        for ele in self.elements:
            orbs.append([u for u in self.orbs[ele] if u!=0]);

        return orbs;

    def generate_irreps(self):
        
        self.irreps1 = {};
        self.irreps2 = {};

        for el in self.elements:

            orb = self.orbs[el];
            string = '';
            for i,n in enumerate(orb):
                if(n!=0):
                    string += str(n) + 'x'+str(i) + 'eo'[i%2] + '+';
            string = string[:-1];
            self.irreps1[el] = o3.Irreps(string);

        for el1, el2 in product(self.elements,self.elements):
            orb1, orb2 = self.orbs[el1], self.orbs[el2];
            string = '';
            for i,m in enumerate(orb1):
                for j,n in enumerate(orb2):
                    if(n!=0 and m!=0):
                        low = abs(i-j);
                        high = i+j;
                        
                        if(i==0 or j==0):
                            string += (str(m*n) + 'x'+str(high) + 'eo'[(i+j)%2] + '+');
                        else:
                            temp = '';
                            for k in range(low,high+1):
                                temp += (str(k) + 'eo'[(i+j)%2] + '+');
                            string += temp*(m*n);
                        
            string = string[:-1];
            self.irreps2[(el1,el2)] = o3.Irreps(string);

        return None;

    def get_pair_irreps(self):
        
        V_irreps = [];
        for i, el1 in enumerate(self.elements):
            for j, el2 in enumerate(self.elements):
                if(j<=i):
                    V_irreps.append(self.irreps2[(el1,el2)]);

        return V_irreps;
    
    def get_onsite_irreps(self):

        irreps = [self.irreps2[(key,key)] for key in self.elements];
        return irreps;

    def get_MO_irreps(self):

        return self.irreps1;
    
    def get_nbasis(self):

        nbasis = {};
        for key in self.orbs:
            orb = self.orbs[key];
            nbasis[key] = sum([(2*i+1)*u for i,u in enumerate(orb)])

        return nbasis;
    
    def get_onehot(self):

        onehot = {};
        for i,el in enumerate(self.elements):
            onehot[el] = [0]*len(self.elements);
            onehot[el][i] = 1;

        return onehot;

    def get_ne(self):

        ne = {key:self.el_dict[key] for key in self.orbs};

        return ne;
    
    def get_hidden_irreps(self, width=8, depth=3, lmax=4):

        irreps_hidden = [];
        w = str(width);
        irreps0 = ['x'+str(i)+m for i, m in product(range(lmax+1), ['e','o'])];
        for i in range(depth):
            if(i==0):
                string = w+'x0e + '+w+'x1o + '+w+'x2e';
            elif(i==1):
                string = "";
                for ir in irreps0[:6]:
                    string += w+ir + ' + ';
                string = string[:-3];
            else:
                string = "";
                for ir in irreps0:
                    string += w+ir + ' + ';
                string = string[:-3];
            irreps_hidden.append(o3.Irreps(string));
        
        return irreps_hidden;
    
    def get_sh_irreps(self, lmax=2):

        return o3.Irreps.spherical_harmonics(lmax=lmax);
    
    def get_input_irreps(self):

        return len(self.elements);
