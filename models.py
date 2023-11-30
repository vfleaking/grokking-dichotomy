import torch
from torch import nn


class DiagonalNet(nn.Module):
    def __init__(self, alpha, L, dimD):
        super().__init__()
        self.u = nn.Parameter(alpha / ((dimD * 2) ** 0.5) * torch.ones(dimD))
        self.v = nn.Parameter(alpha / ((dimD * 2) ** 0.5) * torch.ones(dimD))
        self.L = L
    
    def get_w(self):
        return self.u ** self.L - self.v ** self.L
    
    def forward(self, x):
        return (x @ self.get_w()).unsqueeze(-1)

class HomoMLP(nn.Module):
    def __init__(self, init_scale, L, dimD, dimH, dimO, first_layer_bias, init_method='he'):
        super().__init__()
        self.L = L
        self.dimD = dimD
        self.dimH = dimH
        self.dimO = dimO
        
        self.layers = []
        seq = []
        dimLast = self.dimD
        for k in range(self.L):
            dimNext = self.dimH if k < self.L - 1 else self.dimO
            l = nn.Linear(dimLast, dimNext, bias=(k == 0 and first_layer_bias))
            self.layers.append(l)
            seq.append(l)
            dimLast = dimNext
            if k < self.L - 1:
                seq.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*seq)

        if init_method == 'he':
            for i, l in enumerate(self.layers):
                if i < self.L - 1:
                    torch.nn.init.kaiming_normal_(l.weight.data, nonlinearity='relu')
                else:
                    torch.nn.init.kaiming_normal_(l.weight.data, nonlinearity='linear')
            
                if l.bias is not None:
                    l.bias.data.zero_()

        for i, l in enumerate(self.layers):
            l.weight.data.mul_(init_scale)
    
    def forward(self, x):
        return self.net(x)
    
class MatrixFactorization(nn.Module):
    def __init__(self, alpha, dimD):
        super().__init__()
        self.dimD = dimD

        self.U = nn.Parameter(alpha * torch.eye(dimD))
        self.V = nn.Parameter(alpha * torch.eye(dimD))
        
    def forward(self, x1, x2):
        M = self.U @ self.U.T - self.V @ self.V.T
        return M[x1, x2]
