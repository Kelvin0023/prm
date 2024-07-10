import torch


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    if isinstance(x, torch.Tensor):
        x = x.to(dtype=dtype, device=device)
        x = x.clone().detach().requires_grad_(requires_grad)
    else:
        x = torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
    return x
