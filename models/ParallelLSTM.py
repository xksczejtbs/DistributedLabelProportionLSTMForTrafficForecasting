import torch
from torch import nn
class ParallelLSTM(torch.nn.Module):#IMVTensorLSTM
    __constants__ = ["n_units", "input_dim"]

    def __init__(self, input_dim, output_dim, n_units, device, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)
        self.F_beta = nn.Linear(2 * n_units, 1)
        self.Phi = nn.Linear(2 * n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
        self.device = device


    def forward(self, x):
        #print(x.shape)
        x = torch.transpose(x, 0, 1)
        #print(x.shape)
        x = torch.unsqueeze(x,0)
        #print(x.shape)
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units)
        h_tilda_t = h_tilda_t.to(self.device)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units)
        c_tilda_t = c_tilda_t.to(self.device)
        #outputs = torch.jit.annotate(List[Tensor], [])
        outputs = []
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_o) + self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t * torch.tanh(c_tilda_t))
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        #print("outputshape",outputs.shape)
        outputs = outputs.permute(1, 0, 2, 3)
        #print("outputshape",outputs.shape)
        outputs = torch.squeeze(outputs, 0)
        if self.n_units == 1:
            outputs = torch.squeeze(outputs, 2)
        else:
            print("Warning: n_units != 1. outputs has different shape")
        #print("outputshape", outputs.shape)
        outputs = torch.transpose(outputs, 0, 1)
        #print("outputshape", outputs.shape)
        return outputs
