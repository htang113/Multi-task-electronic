import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3
from e3nn import nn

class V_theta(torch.nn.Module):
    
    def __init__(self, device, emb_neurons: int = 16, scaling=0.2) -> None:
        super().__init__()
        
        # Initialize a Equivariance graph convolutional neural network
        # nodes with distance smaller than max_radius are connected by bonds
        # num_basis is the number of basis for edge feature embedding
        
        self.device = device;
        self.scaling = scaling;
        self.Irreps_HH = [["4x0e","2x1o"],
                         ["2x1o","1x0e+1x1e+1x2e"]];
        
        self.Irreps_CC = [["9x0e","6x1o","3x2e"],                         
                         ["6x1o",("1x0e+1x1e+1x2e+"*4)[:-1],
                          ("1x1o+1x2o+1x3o+"*2)[:-1]],           
                         ["3x2e",("1x1o+1x2o+1x3o+"*2)[:-1],
                          "1x0e+1x1e+1x2e+1x3e+1x4e" 
                          ]];
        
        self.Irreps_CH = [["6x0e","3x1o"],                         
                         ["4x1o",("1x0e+1x1e+1x2e+"*2)[:-1]],           
                         ["2x2e","1x1o+1x2o+1x3o"]];
        
        out = "";
        for f in self.Irreps_HH:
            for f1 in f:
                out += f1+'+';
        self.Irreps_HH = o3.Irreps(out[:-1]);
        out = "";
        for f in self.Irreps_CC:
            for f1 in f:
                out += f1+'+';
        self.Irreps_CC = o3.Irreps(out[:-1]);
        out = "";
        for f in self.Irreps_CH:
            for f1 in f:
                out += f1+'+';
        self.Irreps_CH = o3.Irreps(out[:-1]);
        
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2);
        self.irreps_input = o3.Irreps("2x0e");
        irreps_mid1 = o3.Irreps("8x0e + 8x1o + 8x2e");
        irreps_mid2 = o3.Irreps("8x0e + 8x0o + 8x1e + 8x1o + 8x2e + 8x2o");
        
        self.linear1 = o3.Linear(self.irreps_input,
                                 self.irreps_input);
        self.activation1 = nn.Activation(self.irreps_input, [torch.tanh]);
        
        self.tp1 = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_input,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_mid1,
            shared_weights=False
        )
        
        self.activation2 = nn.Activation(irreps_mid1, [torch.tanh,None,None]);
        self.linear2 = o3.Linear(irreps_mid1,
                                 irreps_mid1);
        
        self.tp2 = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid1,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_mid2,
            shared_weights=False
        )
        
        self.activation3 = nn.Activation(irreps_mid2, [torch.tanh]*2+[None]*4);
        self.linear3 = o3.Linear(irreps_mid2,
                                 irreps_mid2);
        
        self.linearCC = o3.Linear(irreps_mid2,
                                  self.Irreps_CC);
        self.linearHH = o3.Linear(irreps_mid2,
                                  self.Irreps_HH);
        self.linearCH = o3.Linear(irreps_mid2,
                                  self.Irreps_CH);
        
        self.bond_feature = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid2,
            irreps_in2=irreps_mid2,
            irreps_out=irreps_mid2,
            shared_weights=False
        )
        
        self.linearC = o3.Linear(irreps_mid2,
                                 self.Irreps_CC);
        self.linearH = o3.Linear(irreps_mid2,
                                 self.Irreps_HH);
        
        self.screen1 = o3.Linear(irreps_mid2, o3.Irreps("32x0e+1x2e"));
        self.screen_activation = nn.Activation(o3.Irreps("32x0e+1x2e"), [torch.tanh, None]);
        self.screen2 = o3.Linear(o3.Irreps("32x0e+1x2e"), o3.Irreps("1x0e+1x2e"));

        self.gap1 = o3.Linear(irreps_mid2, o3.Irreps("32x0e"));
        self.gap_activation = nn.Activation(o3.Irreps("32x0e"), [torch.tanh]);
        self.gap2 = o3.Linear(o3.Irreps("32x0e"), o3.Irreps("3x0e"));

        self.fc1 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons,emb_neurons, self.tp1.weight_numel], torch.relu);
        self.fc2 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons,emb_neurons, self.tp2.weight_numel], torch.relu);
        self.fc_bond = nn.FullyConnectedNet([1, emb_neurons,emb_neurons,emb_neurons, self.bond_feature.weight_numel], torch.relu);

    def forward(self, minibatch) -> torch.Tensor:
        
        # Forward function of the neural network model
        # positions is a Nx3 tensor, including the N times 3D atomic cartesian coordinate
        # elements is a N-dim list, whose ith component is either 'H' or 'C' denoting 
        # the ith atomic species.
        # The output is a (N*14)x(N*14) V_theta matrix.
        # The i*14+j row/column means the ith atom's jth basis orbital
        # The basis on an atom is ranked as (s1,s2,s3,p1_(-1,0,1),p2_(-1,0,1),d1_(-2,-1,0,1,2))
        sh = minibatch['sh'];
        emb = minibatch['emb'];
        f_in = minibatch['f_in'];
        edge_src = minibatch['edge_src'];
        edge_dst = minibatch['edge_dst'];
        num_nodes = minibatch['num_nodes'];
        num_neighbors = minibatch['num_neighbors'];
        HH_ind = minibatch['HH_ind'];
        CC_ind = minibatch['CC_ind'];
        CH_ind = minibatch['CH_ind'];
        
        node_feature = self.linear1(f_in);
        node_feature = self.activation1(node_feature);
        edge_feature = self.tp1(node_feature[edge_src], sh, self.fc1(emb));
        node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);

        node_feature = self.linear2(node_feature);
        node_feature = self.activation2(node_feature);
        edge_feature = self.tp2(node_feature[edge_src], sh, self.fc2(emb));
        node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
        
        node_feature = self.linear3(node_feature);
        node_feature = self.activation3(node_feature);
        
        HH_feature = self.bond_feature(node_feature[edge_src[HH_ind]], node_feature[edge_dst[HH_ind]],self.fc_bond(emb[HH_ind]));
        CC_feature = self.bond_feature(node_feature[edge_src[CC_ind]], node_feature[edge_dst[CC_ind]],self.fc_bond(emb[CC_ind]));
        CH_feature = self.bond_feature(node_feature[edge_src[CH_ind]], node_feature[edge_dst[CH_ind]],self.fc_bond(emb[CH_ind]));
        
        edge_HH = self.linearHH(self.activation3(HH_feature));
        edge_CC = self.linearCC(self.activation3(CC_feature));
        edge_CH = self.linearCH(self.activation3(CH_feature));
        
        node_C = self.linearC(node_feature);
        node_H = self.linearH(node_feature);

        screen_mat = self.screen2(self.screen_activation(self.screen1(node_feature)));
        gap_mat = [self.gap2(self.gap_activation(self.gap1(CC_feature))), 
                   self.gap2(self.gap_activation(self.gap1(HH_feature))),
                   self.gap2(self.gap_activation(self.gap1(CH_feature)))];

        V_raw = {'H': node_H * self.scaling,
                 'C': node_C * self.scaling,
                 'HH': edge_HH * self.scaling,
                 'CH': edge_CH * self.scaling,
                 'CC': edge_CC * self.scaling,
                 'screen': screen_mat,
                 'gap': gap_mat};
        
        return V_raw;


