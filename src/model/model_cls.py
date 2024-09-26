import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3
from e3nn import nn

class V_theta(torch.nn.Module):
    
    def __init__(self, device, irreps, ele_emb: int=3, emb_neurons: int = 16, scaling=0.2) -> None:
        super().__init__()
        
        # Initialize a Equivariance graph convolutional neural network
        # nodes with distance smaller than max_radius are connected by bonds
        # num_basis is the number of basis for edge feature embedding
        
        self.device = device;
        self.scaling = scaling;
        
        ######## Define the input and output irreps ########
        irreps_pair = irreps.get_pair_irreps();
        irreps_node = irreps.get_onsite_irreps();
        irreps_sh = irreps.get_sh_irreps();
        input_dim = irreps.get_input_irreps();
        irreps_hidden = [o3.Irreps(str(ele_emb)+'x0e')] + irreps.get_hidden_irreps(lmax=2);

        ######### Element embedding network #########
        self.ele_embed = torch.nn.Sequential(
            torch.nn.Linear(input_dim, emb_neurons),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_neurons, emb_neurons),
            torch.nn.Sigmoid(),
            torch.nn.Linear(emb_neurons, ele_emb)
        )

        ######### Define the graph convolutional neural network #########
        tp_convs = [];
        activations= [];
        linears = [];
        fcs = [];
        for i in range(3):
            tp_convs.append(o3.FullyConnectedTensorProduct(
                irreps_in1=irreps_hidden[i],
                irreps_in2=irreps_sh,
                irreps_out=irreps_hidden[i+1],
                shared_weights=False
            ))
            fcs.append(nn.FullyConnectedNet([1, emb_neurons,emb_neurons, 
                            tp_convs[i].weight_numel], torch.tanh));

            irrep = str(irreps_hidden[i+1]);
            Nact, Ntot = irrep.count('0'), irrep.count('+')+1;
            linears.append(o3.FullyConnectedTensorProduct(
                irreps_in1=irreps_hidden[0],
                irreps_in2=irreps_hidden[i+1],
                irreps_out=irreps_hidden[i+1],
            ));

            activations.append(nn.Activation(irreps_hidden[i+1], 
                            [torch.tanh]*Nact+[None]*(Ntot-Nact)));

        ############### Define the bond feature network ###############
        tp_convs.append(o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_hidden[i+1],
            irreps_in2=irreps_hidden[i+1],
            irreps_out=irreps_hidden[i+1],
            shared_weights=False
        ))
        fcs.append(nn.FullyConnectedNet([1, emb_neurons,emb_neurons, 
                        tp_convs[-1].weight_numel], torch.tanh));
        
        self.tp_convs = torch.nn.ModuleList(tp_convs);
        self.fcs = torch.nn.ModuleList(fcs);
        self.linears = torch.nn.ModuleList(linears);
        self.activations = torch.nn.ModuleList(activations);

        self.linear_edge = torch.nn.ModuleList([o3.Linear(irreps_hidden[i+1], 
                                            irreps) for irreps in irreps_pair]);
        self.linear_node = torch.nn.ModuleList([o3.Linear(irreps_hidden[i+1],
                                            irreps) for irreps in irreps_node]);
        
        ########### Define the screen and gap functions ###########

        self.screen1 = o3.Linear(irreps_hidden[i+1], o3.Irreps("32x0e+1x2e"));
        self.screen_activation = nn.Activation(o3.Irreps("32x0e+1x2e"), [torch.tanh, None]);
        self.screen2 = o3.Linear(o3.Irreps("32x0e+1x2e"), o3.Irreps("1x0e+1x2e"));

        self.gap1 = o3.Linear(irreps_hidden[i+1], o3.Irreps("32x0e"));
        self.gap_activation = nn.Activation(o3.Irreps("32x0e"), [torch.tanh]);
        self.gap2 = o3.Linear(o3.Irreps("32x0e"), o3.Irreps("3x0e"));

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
        mask = minibatch['pair_ind'];
        
        inputs = self.ele_embed(f_in);
        node_feature = inputs;

        for i in range(3):
            edge_feature = self.tp_convs[i](node_feature[edge_src], sh, self.fcs[i](emb));
            node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors[:,None]**0.5);
            node_feature = self.linears[i](inputs, node_feature);
            node_feature = self.activations[i](node_feature);

        output_edge_feature = self.tp_convs[-1](node_feature[edge_src], node_feature[edge_dst],self.fcs[-1](emb));
        
        edge_pair = [];
        for i in range(len(self.linear_edge)):
            res = self.linear_edge[i](output_edge_feature[mask[i]])* self.scaling;
            edge_pair.append(res);
        
        node_onsite = [linear_layer(node_feature)* self.scaling for linear_layer in self.linear_node];

        screen_mat = self.screen2(self.screen_activation(self.screen1(node_feature)));
        gap_mat = self.gap2(self.gap_activation(self.gap1(node_feature)));

        V_raw = {'node': node_onsite,
                 'edge': edge_pair,
                 'screen': screen_mat,
                 'gap': gap_mat};
        
        return V_raw;


