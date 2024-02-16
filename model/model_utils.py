import torch
import torch.nn as nn

def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def local_cov(pts, idx):
    batch_size, num_dims, num_points = pts.size()
    x = pts.transpose(2, 1).contiguous()  # (batch_size, num_points, 3)
    x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, 3)

    x = torch.matmul(x[:, :, 0].unsqueeze(3), x[:, :, 1].unsqueeze(
        2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
    # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
    x = x.view(batch_size, num_points, 9).transpose(2, 1)  # (batch_size, 9, num_points)

    x = torch.cat((pts, x), dim=1)  # (batch_size, 12, num_points)

    return x.transpose(1, 2)


def local_maxpool(x, idx):
    batch_size, num_points, num_dims = x.size()

    x = x.reshape(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)  # (batch_size, num_points, num_dims)

    return x

class Residual_Linear_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Residual_Linear_Layer, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_channels, int(out_channels*0.5), bias=bias),
                                    nn.ReLU(),
                                    nn.Linear(int(out_channels*0.5), out_channels, bias=bias))
    
    def forward(self, x):
        return self.linear(x) + x