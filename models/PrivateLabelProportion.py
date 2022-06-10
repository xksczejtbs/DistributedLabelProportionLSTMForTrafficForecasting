import torch
import torch.nn.functional as F
from models.LabelMessagePassing import MPGCNConv
from models.ParallelLSTM import ParallelLSTM
from models.ParallelLinear import ParallelLinear
from models.ParallelLinear import ParallelLinearTransposed
import params

class LabelProportionGCN(torch.nn.Module):
    def __init__(self, node_features, num_sensors, num_node_feature, device):
        # node_features = num_sensors * num_node_features
        super(LabelProportionGCN, self).__init__()
        #for deterministic behavior of lstm set env variable CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:2
        #self.recurrent = LSTM(input_size=node_features, hidden_size=4, num_layers=1, batch_first=True) #assume unbatched
        self.parallelrecurrent = ParallelLSTM(input_dim = node_features, output_dim=1, n_units=1, device=device)
        self.num_node_feature = num_node_feature
        #self.aggre = MPGCNConv(2, 2, device=device)
        #self.linear = torch.nn.Linear(11558, 11558, bias=False)
        self.parallellinear = ParallelLinear(input_dim=num_sensors, n_units=num_node_feature, output_dim=num_node_feature, device=device)

        #torch.nn.init.eye_(self.linear.weight)

    def forward(self, x, edge_index, edge_weight):
        h = self.parallelrecurrent(x)
        h = F.relu(h)
        h = h.view(-1, self.num_node_feature, params.num_timesteps_out)
        h = self.parallellinear(h)
        h = h.reshape(-1, params.num_timesteps_out)
        return h

class LabelProportionToDense(torch.nn.Module):
    def __init__(self, node_features, num_sensors, num_node_feature, epsilon, mins, maxs, device):
        #node_features = num_sensors * num_node_features
        super(LabelProportionToDense, self).__init__()
        self.parallelrecurrent = ParallelLSTM(input_dim = node_features, output_dim=1, n_units=1, device=device)
        self.num_node_feature = num_node_feature
        self.message_passing_model = MPGCNConv(num_node_feature, num_node_feature, num_node_feature, num_sensors, device=device, epsilon=epsilon,
                                          mins=mins, maxs=maxs)
        self.parallellinear = ParallelLinear(input_dim=num_sensors, n_units=num_node_feature, output_dim=num_node_feature, device=device)
        self.parallellineartransposed = ParallelLinearTransposed(input_dim=num_sensors, n_units=(params.num_timesteps_in+params.bins), output_dim=params.num_timesteps_out, device=device)

    def forward(self, x, y, edge_index, edge_weight):
        h = self.parallelrecurrent(x)
        h = F.relu(h)
        h = h.view(-1,self.num_node_feature,params.num_timesteps_out)
        with torch.no_grad():
            histograms = self.message_passing_model(y, edge_index, edge_weight)
        h = torch.cat([h,histograms],axis=2) # assumed shapes: (nodes,features,bins) and (nodes,features,timestamps). Resulting shape (nodes, features, bins+timestamp)
        h = self.parallellinear(h)
        h = self.parallellineartransposed(h)
        h = h.reshape(-1, params.num_timesteps_out) # should be (-1, 12) but how to reduce from 22 to 12
        return h

class LabelProportionLocal(torch.nn.Module):
    def __init__(self, node_features, num_sensors, num_node_feature, epsilon, mins, maxs, device):
        #node_features = num_sensors * num_node_features
        super(LabelProportionLocal, self).__init__()
        self.parallelrecurrent = ParallelLSTM(input_dim = node_features, output_dim=1, n_units=1, device=device)
        self.num_node_feature = num_node_feature
        self.parallellinear = ParallelLinear(input_dim=num_sensors, n_units=num_node_feature, output_dim=num_node_feature, device=device)
        self.parallellineartransposed = ParallelLinearTransposed(input_dim=num_sensors, n_units=(params.num_timesteps_in), output_dim=params.num_timesteps_out, device=device)

    def forward(self, x, edge_index, edge_weight):
        h = self.parallelrecurrent(x)
        h = F.relu(h)
        h = h.view(-1,self.num_node_feature,params.num_timesteps_out)#(nodes,features,timestamps)
        #h = torch.cat([h,histograms],axis=2) # assumed shapes: (nodes,features,bins) and (nodes,features,timestamps). Resulting shape (nodes, features, bins+timestamp)
        h = self.parallellinear(h)
        h = self.parallellineartransposed(h)
        h = h.reshape(-1, params.num_timesteps_out) # should be (-1, 12) but how to reduce from 22 to 12
        return h