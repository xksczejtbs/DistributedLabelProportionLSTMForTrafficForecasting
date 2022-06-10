import torch
from torch import nn

class ParallelLinear(nn.Module):#IMVTensorLSTM
    __constants__ = ["n_units", "input_dim"]
    """ 
        input_dim = edges 
        n_units = 2 (=features speed and traffic density)    
        output_dim = n_units
    """
    def __init__(self, input_dim, n_units, output_dim, device):
        super().__init__()
        self.cpu = torch.device("cpu")
        self.linears = nn.ModuleList([nn.Linear(n_units, output_dim) for _ in range(input_dim)])
        for t in range(input_dim):
            torch.nn.init.eye_(self.linears[t].weight)
            torch.nn.init.zeros_(self.linears[t].bias)
            self.linears[t] = self.linears[t].to(self.cpu)#device)
        self.n_units = n_units
        self.input_dim = input_dim
        self.device = device


    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = x.to(self.cpu)
        outputs = []
        #thinking about let cpu do this work because it is not parallelisable
        # idea: sparse matrix multiplication https://pytorch.org/docs/stable/generated/torch.sparse.mm.html
        for t in range(x.shape[0]): # edges are stacked
            self.linears[t] = self.linears[t].to(self.cpu)
            h = self.linears[t](x[t])
            outputs += [h]
        outputs = torch.stack(outputs)
        outputs = torch.transpose(outputs, 1, 2)
        return outputs.to(self.device)

class ParallelLinearTransposed(nn.Module):  # IMVTensorLSTM
    __constants__ = ["n_units", "input_dim"]
    """ 
        input_dim = edges 
        n_units = 2 (=features speed and traffic density)    
        output_dim = n_units
    """

    def __init__(self, input_dim, n_units, output_dim, device):
        super().__init__()
        self.cpu = torch.device("cpu")
        self.linears = nn.ModuleList([nn.Linear(n_units, output_dim) for _ in range(input_dim)])
        for t in range(input_dim):
            torch.nn.init.eye_(self.linears[t].weight)
            torch.nn.init.zeros_(self.linears[t].bias)
            self.linears[t] = self.linears[t].to(self.cpu)  # device)
        self.n_units = n_units
        self.input_dim = input_dim
        self.device = device

    def forward(self, x):
        #x = torch.transpose(x, 1, 2)#deleted
        x = x.to(self.cpu)
        outputs = []
        # thinking about let cpu do this work because it is not parallelisable
        # idea: sparse matrix multiplication https://pytorch.org/docs/stable/generated/torch.sparse.mm.html
        for t in range(x.shape[0]):  # edges are stacked
            self.linears[t] = self.linears[t].to(self.cpu)
            h = self.linears[t](x[t])
            outputs += [h]
        outputs = torch.stack(outputs)
        #outputs = torch.transpose(outputs, 1, 2) #deleted
        return outputs.to(self.device)

if __name__ == "__main__":
    edges = 5000
    features = 2
    timestamps = 12
    x = torch.randn(edges, features, timestamps)
    print("x",x.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParallelLinear(edges, features, features, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    y = model(x)
    print("y",y.shape)
    y_actual = torch.randn(edges, features, timestamps)
    loss = torch.mean(y-y_actual)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()