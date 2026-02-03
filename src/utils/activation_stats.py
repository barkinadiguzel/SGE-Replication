import torch

def group_variance(lengths: torch.Tensor):
    var = lengths.var(dim=(0,2,3), unbiased=False)  
    return var  

def group_mean(lengths: torch.Tensor):
    return lengths.mean(dim=(0,2,3))

def sparsity_ratio(lengths: torch.Tensor, thresh=1e-3):
    total = lengths.numel() / lengths.shape[1] 
    zeros = (lengths <= thresh).sum(dim=(0,2,3)).float()
    return (zeros / total)  
