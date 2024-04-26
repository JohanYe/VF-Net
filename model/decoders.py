import itertools
from math import prod
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from model.model_utils import Residual_Linear_Layer, local_cov, knn


class Decoder_Block(nn.Module):
    """ basic block with inits and build grid """

    def __init__(self, args, num_points):
        super(Decoder_Block, self).__init__()
        self.num_points_sqrt = int(np.sqrt(num_points))
        self.m = self.num_points_sqrt * self.num_points_sqrt
        self.shape = args.fold_orig_shape
        self.feat_dims = args.feat_dims
        self.point_encoding = True if args.model == "vae" else args.point_encoding

        if args.fold_orig_shape == "plane":
            self.meshgrid = [[-1, 1, self.num_points_sqrt], [-1, 1, self.num_points_sqrt]]
        elif args.fold_orig_shape == "sphere":
            self.sphere = self.sample_spherical(num_points)

        self.grid = None
        self.std = None

    def sample_spherical(self, npoints, ndim=3):
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    def build_grid(self, batch_size, sample_grid, device, edge_only=False):
        if self.shape == 'plane':
            if not sample_grid:  # first because most often used.
                x = np.linspace(*self.meshgrid[0])
                y = np.linspace(*self.meshgrid[1])
                points = np.array(list(itertools.product(x, y)))
            else:  # put here, because only used in visualization
                if not edge_only:
                    # x = np.random.uniform(self.meshgrid[0][0], self.meshgrid[0][1], int(self.meshgrid[0][-1]))
                    # y = np.random.uniform(self.meshgrid[0][0], self.meshgrid[0][1], int(self.meshgrid[1][-1]))
                    # points = np.array(list(itertools.product(x, y)))
                    points = np.random.uniform(-1, 1, (self.m, 2))
                else:
                    edge_points = []
                    x = np.linspace(*self.meshgrid[0])
                    y = np.linspace(*self.meshgrid[1])
                    # here twice, in case mesh is uneven
                    for i in range(self.meshgrid[0][-1]):  # over x
                        edge_points.append([x[i], y.min()])
                        edge_points.append([x[i], y.max()])
                    for i in range(self.meshgrid[1][-1]):  # over y
                        edge_points.append([x.min(), y[i]])
                        edge_points.append([x.max(), y[i]])
                    points = np.array(edge_points)

        elif self.shape == 'sphere':
            points = self.sphere if not sample_grid else self.sample_spherical(
                self.num_points_sqrt * self.num_points_sqrt)
        elif self.shape == 'gaussian':
            points = self.gaussian
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float().to(device)


class Decoder_Linear(Decoder_Block):
    """ Linear Decoder """

    def __init__(self, args, num_points):
        super(Decoder_Linear, self).__init__(args, num_points)

        self.folding1 = nn.Sequential(
            nn.Linear(int(args.feat_dims) + 2 + (self.shape != 'plane'), args.feat_dims),
            nn.ReLU(),
            Residual_Linear_Layer(args.feat_dims, args.feat_dims),
            nn.ReLU(),
            Residual_Linear_Layer(args.feat_dims, args.feat_dims),
            nn.ReLU(),
            nn.Linear(args.feat_dims, 3),
        )
        self.folding2 = nn.Sequential(
            nn.Linear(args.feat_dims + 3, args.feat_dims),
            nn.ReLU(),
            Residual_Linear_Layer(args.feat_dims, args.feat_dims),
            nn.ReLU(),
            Residual_Linear_Layer(args.feat_dims, args.feat_dims),
            nn.ReLU(),
            nn.Linear(args.feat_dims, 3),
        )

        if self.point_encoding:
            self.orig_dims = 3 + 3 * args.point_normals
            self.grid_map = nn.Sequential(nn.Linear(args.feat_dims + 128, int(args.feat_dims / 2)),
                                          nn.ReLU(),
                                          nn.Linear(int(args.feat_dims / 2), int(args.feat_dims / 4)),
                                          nn.ReLU(),
                                          nn.Linear(int(args.feat_dims / 4), 2),
                                          nn.Tanh())

    def init_std(self, device):
        # from tooth_wear_trainer import PointNet
        # self.std = PointNetEncoder().to(device)
        self.std = variance_network(512, device)

    def decode(self, x, grid):
        cat1 = torch.cat((x, grid),
                         dim=-1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)

        folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
        cat2 = torch.cat((x, folding_result1), dim=-1)  # (batch_size, 515, num_points)
        folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
        if self.std is None:
            return {"reconstruction": folding_result2.transpose(1, 2)}  # (batch_size, num_points ,3)
        else:
            # std = self.std(folding_result2.transpose(1, 2))
            std = self.std(x, folding_result2)
            return {"reconstruction": folding_result2.transpose(1, 2),
                    "std":std}  # (batch_size, num_points ,3)


    def forward(self, x, pts_orig, sample_grid=False, edge_only=False):

        if self.point_encoding:
            assert pts_orig is not None
            x = x.repeat(1, pts_orig.shape[1], 1) # (batch_size, num_points, latent_dims)
            cat = torch.cat((x, pts_orig), dim=-1)  #  (batch_size, num_points, latent_dims+local_feat_dims)
            self.grid = self.grid_map(cat) # (batch_size, num_points, 2)
            return self.decode(x, self.grid)  # (batch_size, num_points ,3)
        else:
            m = self.m if not edge_only else int(self.meshgrid[0][-1]) * 2 + int(self.meshgrid[1][-1]) * 2
            x = x.repeat(1, m, 1)  # (batch_size, feat_dims, num_points)

            if self.grid is None or x.shape[0] != self.grid.shape[
                0]:  # doesn't exist or sample or batch_size has changed
                self.grid = self.build_grid(x.shape[0], sample_grid, x.device, edge_only)
                if self.grid.shape[1] < 4:
                    self.grid = self.grid.transpose(1, 2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)

            output = self.decode(x, self.grid)  # (batch_size, num_points ,3)

            if sample_grid:
                self.grid = None

            return output
        

class variance_network(nn.Module):
    def __init__(self, feat_dims, device):
        super(variance_network, self).__init__()
        self.feat_dims = feat_dims
        self.device = device

        self.std_1 = nn.Sequential(
            nn.Linear(self.feat_dims + 3 + 9, self.feat_dims),
            nn.ReLU(),
            Residual_Linear_Layer(self.feat_dims, self.feat_dims),
            nn.ReLU(),
            nn.Linear(self.feat_dims, 1),
        ).to(device)

        self.std_2 = nn.Sequential(
            nn.Linear(self.feat_dims + 1, self.feat_dims),
            nn.ReLU(),
            Residual_Linear_Layer(self.feat_dims, self.feat_dims),
            nn.ReLU(),
            nn.Linear(self.feat_dims, 1),
        ).to(device)

        self.std_3 = nn.Sequential(
            nn.Linear(self.feat_dims + 1, self.feat_dims),
            nn.ReLU(),
            Residual_Linear_Layer(self.feat_dims, self.feat_dims),
            nn.ReLU(),
            nn.Linear(self.feat_dims, 1),
        ).to(device)


    def forward(self, latent, recon_pc):
        recon_pc = recon_pc.transpose(1, 2)
        idx = knn(recon_pc, k=16)
        out = local_cov(recon_pc, idx)  # (batch_size, 3, num_points) -> (batch_size, 12, num_points])
        recon_pc = recon_pc.transpose(1, 2)
        cat = torch.cat((latent, out), dim=-1)
        out = self.std_1(cat)
        cat2 = torch.cat((latent, out), dim=-1)
        out = out + self.std_2(cat2)
        cat3 = torch.cat((latent,out), dim=-1)
        return out + self.std_3(cat3)
