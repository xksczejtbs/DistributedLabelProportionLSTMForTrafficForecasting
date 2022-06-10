from torch_geometric.nn import MessagePassing
import torch
from torch_geometric.utils import add_self_loops, degree
from utils import histogram_stacked
import params
class MPGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_feature, num_sensors, device, epsilon, mins, maxs):
        super().__init__(aggr='add')  # aggregation (Step 5).
        self.num_node_feature = num_node_feature
        self.num_sensors = num_sensors
        self.device = device
        self.epsilon = epsilon
        self.mins = mins
        self.maxs = maxs
    @torch.no_grad()
    def forward(self, x, edge_index, edge_weight):
        x = x.view(-1, self.num_node_feature, params.num_timesteps_in)
        x = histogram_stacked(x, bins=params.bins, mins=self.mins, maxs=self.maxs)/x.shape[0]
        sensitivity = 1.0 # because of histogram
        if self.epsilon > 0: #otherwise do not add noise
            with torch.no_grad():
                var = 1.0 / (self.epsilon * x.shape[0])
                gaussian = False
                if gaussian:
                    #gaussian noise with zero mean and var 1/epsilon
                    x = x + (var**0.5) * torch.randn(x.shape, requires_grad=False, device=self.device)
                else: # laplace
                    noise_distribution = torch.distributions.laplace.Laplace(0, var)
                    x = x + noise_distribution.sample(x.shape)
        x = x.view(self.num_sensors,self.num_node_feature*params.bins)
        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm).view(self.num_sensors,self.num_node_feature,params.bins)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
