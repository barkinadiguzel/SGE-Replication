import torch
import torch.nn as nn

class SGE(nn.Module):
    def __init__(self, groups=64, eps=1e-5):
        super().__init__()
        self.groups = groups
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(groups))
        self.beta = nn.Parameter(torch.zeros(groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        assert C % self.groups == 0, "C must be divisible by groups"
        Cg = C // self.groups

        xg = x.view(N, self.groups, Cg, H, W)

        g = xg.mean(dim=(3,4))  

        g_exp = g.unsqueeze(-1).unsqueeze(-1)  
        c = (g_exp * xg).sum(dim=2)            

        mu = c.mean(dim=(2,3), keepdim=True)
        sigma = c.std(dim=(2,3), unbiased=False, keepdim=True)
        c_hat = (c - mu) / (sigma + self.eps)

        gamma = self.gamma.view(1, self.groups, 1, 1)
        beta = self.beta.view(1, self.groups, 1, 1)
        a = gamma * c_hat + beta

        s = torch.sigmoid(a)  

       
        s_exp = s.unsqueeze(2)   
        xg_hat = xg * s_exp


        out = xg_hat.view(N, C, H, W)
        return out

