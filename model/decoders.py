import itertools
from math import prod
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from stochman import nnj
from model.foldingnet.foldingnet_utils import Residual_Linear_Layer, local_cov, knn


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

        self.folding1 = nnj.Sequential(
            nnj.Linear(int(args.feat_dims) + 2 + (self.shape != 'plane'), args.feat_dims),
            nnj.ReLU(),
            Residual_Linear_Layer(args.feat_dims, args.feat_dims),
            nnj.ReLU(),
            Residual_Linear_Layer(args.feat_dims, args.feat_dims),
            nnj.ReLU(),
            nnj.Linear(args.feat_dims, 3),
        )
        self.folding2 = nnj.Sequential(
            nnj.Linear(args.feat_dims + 3, args.feat_dims),
            nnj.ReLU(),
            Residual_Linear_Layer(args.feat_dims, args.feat_dims),
            nnj.ReLU(),
            Residual_Linear_Layer(args.feat_dims, args.feat_dims),
            nnj.ReLU(),
            nnj.Linear(args.feat_dims, 3),
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

    def decode(self, x, grid, jacobian=False):
        cat1 = torch.cat((x, grid),
                         dim=-1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)

        if not jacobian:
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

        else:
            #  jacobi hack stuff:
            cat1 = cat1.view(-1, x.shape[-1] + 2)  # [num_points, latent_dim, bs] for memory
            jac_init = torch.eye(prod(cat1.shape[1:]), 2, dtype=x.dtype, device=x.device).flip(dims=(0, 1)).repeat(
                cat1.shape[0], 1, 1).reshape(cat1.shape[0], cat1.shape[-1], 2)

            folding_result1, jac1 = self.folding1(cat1, jacobian=jac_init)  # (batch_size, 3, num_points)

            #  more complicated jacobi hack stuff:
            cat2 = torch.cat((x.view(-1, x.shape[-1]), folding_result1),
                             dim=1)  # [num_points, latent_dim, bs] for memory
            jac1_cat = torch.cat(
                (torch.zeros(jac1.shape[0], x.shape[-1], *jac1.shape[2:], device=jac1.device), jac1), dim=1)

            folding_result2, jac2 = self.folding2(cat2, jacobian=jac1_cat)  # (batch_size, 3, num_points)
            if self.std is None:
                return {"reconstruction": folding_result2.unsqueeze(1),
                        "jacobian": jac2,
                        "jacobian_fold1": jac1,
                        "fold1": folding_result1}  # (batch_size, num_points , 3), (batch_size, num_points , 3, 2)
            else:
                std = self.std(cat2)
                return {"reconstruction": folding_result2.unsqueeze(1),
                        "std": std,
                        "jacobian": jac2,
                        "jacobian_fold1": jac1,
                        "fold1": folding_result1}  # (batch_size, num_points , 3), (batch_size, num_points , 3, 2)

    def forward(self, x, pts_orig, sample_grid=False, edge_only=False, jacobian=False):

        if self.point_encoding:
            assert pts_orig is not None
            x = x.repeat(1, pts_orig.shape[1], 1) # (batch_size, num_points, latent_dims)
            cat = torch.cat((x, pts_orig), dim=-1)  #  (batch_size, num_points, latent_dims+local_feat_dims)
            self.grid = self.grid_map(cat) # (batch_size, num_points, 2)
            return self.decode(x, self.grid, jacobian=jacobian)  # (batch_size, num_points ,3)
        else:
            m = self.m if not edge_only else int(self.meshgrid[0][-1]) * 2 + int(self.meshgrid[1][-1]) * 2
            x = x.repeat(1, m, 1)  # (batch_size, feat_dims, num_points)

            if self.grid is None or x.shape[0] != self.grid.shape[
                0]:  # doesn't exist or sample or batch_size has changed
                self.grid = self.build_grid(x.shape[0], sample_grid, x.device, edge_only)
                if self.grid.shape[1] < 4:
                    self.grid = self.grid.transpose(1, 2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)

            output = self.decode(x, self.grid, jacobian=jacobian)  # (batch_size, num_points ,3)

            if sample_grid:
                self.grid = None

            return output


class Decoder_Conv(Decoder_Block):
    """ Convolutional Decoder """

    def __init__(self, args, num_points):
        super(Decoder_Conv, self).__init__(args, num_points)

        self.folding1 = nnj.Sequential(
            nnj.Conv1d(args.feat_dims + 2 + (self.shape != 'plane'), args.feat_dims, 1),
            nnj.ReLU(),
            nnj.Conv1d(args.feat_dims, args.feat_dims, 1),
            nnj.ReLU(),
            nnj.Conv1d(args.feat_dims, args.feat_dims, 1),
            nnj.ReLU(),
            nnj.Conv1d(args.feat_dims, 3, 1),
        )
        self.folding2 = nnj.Sequential(
            nnj.Conv1d(args.feat_dims + 3, args.feat_dims, 1),
            nnj.ReLU(),
            nnj.Conv1d(args.feat_dims, args.feat_dims, 1),
            nnj.ReLU(),
            nnj.Conv1d(args.feat_dims, args.feat_dims, 1),
            nnj.ReLU(),
            nnj.Conv1d(args.feat_dims, 3, 1),
        )

        if self.point_encoding:
            self.orig_dims = 3 + 3 * args.point_normals
            self.grid_map = nn.Sequential(nn.Conv1d(args.feat_dims + 64, int(args.feat_dims / 2), 1),
                                          nn.ReLU(),
                                          nn.Conv1d(int(args.feat_dims / 2), int(args.feat_dims / 4), 1),
                                          nn.ReLU(),
                                          nn.Conv1d(int(args.feat_dims / 4), 2, 1),
                                          nn.Tanh())

    def init_std(self, device):
        self.std = nnj.Sequential(
            nnj.Conv1d(self.feat_dims + 3, self.feat_dims, 1),
            nnj.ReLU(),
            nnj.Conv1d(self.feat_dims, self.feat_dims, 1),
            nnj.ReLU(),
            nnj.Conv1d(self.feat_dims, self.feat_dims, 1),
            nnj.ReLU(),
            nnj.Conv1d(self.feat_dims, 3, 1),
        ).to(device)

    def decode(self, x, grid, jacobian=False):
        cat1 = torch.cat((x, grid), dim=1)  # (batch_size, feat_dims+2, num_points)

        if not jacobian:
            folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
            cat2 = torch.cat((x, folding_result1), dim=1)  # (batch_size, 515, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
            return {"reconstruction": folding_result2.transpose(1, 2)}  # (batch_size, num_points ,3)
        else:
            #  jacobi hack stuff:
            cat1 = cat1.view(-1, x.shape[-1] + 2)  # [num_points, latent_dim, bs] for memory
            jac_init = torch.eye(prod(cat1.shape[1:]), 2, dtype=x.dtype, device=x.device).flip(dims=(0, 1)).repeat(
                cat1.shape[0], 1, 1).reshape(cat1.shape[0], cat1.shape[-1], 2)

            folding_result1, jac1 = self.folding1(cat1, jacobian=jac_init)  # (batch_size, 3, num_points)

            #  more complicated jacobi hack stuff:
            cat2 = torch.cat((x.view(-1, x.shape[-1]), folding_result1),
                             dim=1)  # [num_points, latent_dim, bs] for memory
            jac1_cat = torch.cat(
                (torch.zeros(jac1.shape[0], x.shape[-1], *jac1.shape[2:], device=jac1.device), jac1), dim=1)

            folding_result2, jac2 = self.folding2(cat2, jacobian=jac1_cat)  # (batch_size, 3, num_points)
            if self.std is None:
                return {"reconstruction": folding_result2.unsqueeze(1),
                        "jacobian": jac2,
                        "jacobian_fold1": jac1,
                        "fold1": folding_result1}  # (batch_size, num_points , 3), (batch_size, num_points , 3, 2)
            else:
                std = self.std(cat2)
                return {"reconstruction": folding_result2.unsqueeze(1),
                        "std": std,
                        "jacobian": jac2,
                        "jacobian_fold1": jac1,
                        "fold1": folding_result1}  # (batch_size, num_points , 3), (batch_size, num_points , 3, 2)

    def forward(self, x, pts_orig, sample_grid=False, edge_only=False, jacobian=False):

        if self.point_encoding:
            assert pts_orig is not None
            x = x.transpose(1,2).repeat(1, 1, pts_orig.shape[-1]) # (batch_size, latent_dims, num_points)
            cat = torch.cat((x, pts_orig), dim=1)  # (batch_size, latent_dims + local_feat_dims, num_points)
            self.grid = self.grid_map(cat)  # (batch_size, 2, num_points)
            return self.decode(x, self.grid, jacobian=jacobian)  # (batch_size, num_points ,3)
        else:
            m = self.m if not edge_only else int(self.meshgrid[0][-1]) * 2 + int(self.meshgrid[1][-1]) * 2
            x = x.repeat(1, m, 1)  # (batch_size, feat_dims, num_points)

            if self.grid is None or x.shape[0] != self.grid.shape[
                0]:  # doesn't exist or sample or batch_size has changed
                self.grid = self.build_grid(x.shape[0], sample_grid, x.device, edge_only)
                if self.grid.shape[1] < 4:
                    self.grid = self.grid.transpose(1, 2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)

            output = self.decode(x, self.grid, jacobian=jacobian)  # (batch_size, num_points ,3)

            if sample_grid:
                self.grid = None

            return output


class FoldNet_Decoder_stochman(nn.Module):
    def __init__(self, args, num_points):
        super(FoldNet_Decoder_stochman, self).__init__()
        self.num_points_sqrt = int(np.sqrt(num_points))
        self.m = self.num_points_sqrt * self.num_points_sqrt
        self.shape = args.fold_orig_shape
        self.feat_dims = args.feat_dims
        self.point_encoding = True if args.model == "vae" else args.point_encoding

        if args.fold_orig_shape == "plane":
            self.meshgrid = [[-1, 1, self.num_points_sqrt], [-1, 1, self.num_points_sqrt]]
        elif args.fold_orig_shape == "sphere":
            self.sphere = self.sample_spherical(num_points)

        self.folding1 = nnj.Sequential(
            nnj.Linear(args.feat_dims + 2 + (self.shape != 'plane'), args.feat_dims),
            nnj.ReLU(),
            nnj.Linear(args.feat_dims, args.feat_dims),
            nnj.ReLU(),
            nnj.Linear(args.feat_dims, args.feat_dims),
            nnj.ReLU(),
            nnj.Linear(args.feat_dims, 3),
        )
        self.folding2 = nnj.Sequential(
            nnj.Linear(args.feat_dims + 3, args.feat_dims),
            nnj.ReLU(),
            nnj.Linear(args.feat_dims, args.feat_dims),
            nnj.ReLU(),
            nnj.Linear(args.feat_dims, args.feat_dims),
            nnj.ReLU(),
            nnj.Linear(args.feat_dims, 3),
        )
        self.grid = None
        self.std = None

        if self.point_encoding:
            self.orig_dims = 3 + 3 * args.point_normals
            self.grid_map = nn.Sequential(nn.Linear(args.feat_dims + 64, int(args.feat_dims / 2)),
                                          nn.ReLU(),
                                          nn.Linear(int(args.feat_dims / 2), int(args.feat_dims / 4)),
                                          nn.ReLU(),
                                          nn.Linear(int(args.feat_dims / 4), 2),
                                          nn.Tanh())

    def init_std(self, device):
        self.std = variance_network(512, device)

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

    def decode(self, x, grid, jacobian=False):
        cat1 = torch.cat((x, grid),
                         dim=-1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)

        if not jacobian:
            folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
            cat2 = torch.cat((x, folding_result1), dim=-1)  # (batch_size, 515, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
            return {"reconstruction": folding_result2.transpose(1, 2)}  # (batch_size, num_points ,3)
        else:
            #  jacobi hack stuff:
            cat1 = cat1.view(-1, x.shape[-1] + 2)  # [num_points, latent_dim, bs] for memory
            jac_init = torch.eye(prod(cat1.shape[1:]), 2, dtype=x.dtype, device=x.device).flip(dims=(0, 1)).repeat(
                cat1.shape[0], 1, 1).reshape(cat1.shape[0], cat1.shape[-1], 2)

            folding_result1, jac1 = self.folding1(cat1, jacobian=jac_init)  # (batch_size, 3, num_points)

            #  more complicated jacobi hack stuff:
            cat2 = torch.cat((x.view(-1, x.shape[-1]), folding_result1),
                             dim=1)  # [num_points, latent_dim, bs] for memory
            jac1_cat = torch.cat(
                (torch.zeros(jac1.shape[0], x.shape[-1], *jac1.shape[2:], device=jac1.device), jac1), dim=1)

            folding_result2, jac2 = self.folding2(cat2, jacobian=jac1_cat)  # (batch_size, 3, num_points)
            if self.std is None:
                return {"reconstruction": folding_result2.unsqueeze(1),
                        "jacobian": jac2,
                        "jacobian_fold1": jac1,
                        "fold1": folding_result1}  # (batch_size, num_points , 3), (batch_size, num_points , 3, 2)
            else:
                std = self.std(cat2)
                return {"reconstruction": folding_result2.unsqueeze(1),
                        "std": std,
                        "jacobian": jac2,
                        "jacobian_fold1": jac1,
                        "fold1": folding_result1}  # (batch_size, num_points , 3), (batch_size, num_points , 3, 2)

    def forward(self, latent_codes, feat1, sample_grid=False, edge_only=False, jacobian=False):

        if self.point_encoding:
            assert feat1 is not None
            latent_codes = latent_codes.repeat(1, feat1.shape[1], 1)
            cat = torch.cat((latent_codes, feat1), dim=-1)
            self.grid = self.grid_map(cat)
            return self.decode(latent_codes, self.grid, jacobian=jacobian)  # (batch_size, num_points ,3)
        else:
            m = self.m if not edge_only else int(self.meshgrid[0][-1]) * 2 + int(self.meshgrid[1][-1]) * 2
            latent_codes = latent_codes.repeat(1, m, 1)  # (batch_size, feat_dims, num_points)

            if self.grid is None or latent_codes.shape[0] != self.grid.shape[
                0]:  # doesn't exist or sample or batch_size has changed
                self.grid = self.build_grid(latent_codes.shape[0], sample_grid, latent_codes.device, edge_only)
                if self.grid.shape[1] < 4:
                    self.grid = self.grid.transpose(1, 2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)

            output = self.decode(latent_codes, self.grid, jacobian=jacobian)  # (batch_size, num_points ,3)

            if sample_grid:
                self.grid = None

            return output


class variance_network(nn.Module):
    def __init__(self, feat_dims, device):
        super(variance_network, self).__init__()
        self.feat_dims = feat_dims
        self.device = device

        self.std_1 = nnj.Sequential(
            nnj.Linear(self.feat_dims + 3 + 9, self.feat_dims),
            nnj.ReLU(),
            Residual_Linear_Layer(self.feat_dims, self.feat_dims),
            nnj.ReLU(),
            nnj.Linear(self.feat_dims, 1),
        ).to(device)

        self.std_2 = nnj.Sequential(
            nnj.Linear(self.feat_dims + 1, self.feat_dims),
            nnj.ReLU(),
            Residual_Linear_Layer(self.feat_dims, self.feat_dims),
            nnj.ReLU(),
            nnj.Linear(self.feat_dims, 1),
        ).to(device)

        self.std_3 = nnj.Sequential(
            nnj.Linear(self.feat_dims + 1, self.feat_dims),
            nnj.ReLU(),
            Residual_Linear_Layer(self.feat_dims, self.feat_dims),
            nnj.ReLU(),
            nnj.Linear(self.feat_dims, 1),
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


class STN3d(nn.Module):
    def __init__(self, channels, channel):
        super(STN3d, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv1d(channel, self.channels[0], 1)
        self.conv2 = torch.nn.Conv1d(self.channels[0], self.channels[1], 1)
        self.conv3 = torch.nn.Conv1d(self.channels[1], self.channels[2], 1)
        self.bn1 = nn.BatchNorm1d(self.channels[0])
        self.bn2 = nn.BatchNorm1d(self.channels[1])
        self.bn3 = nn.BatchNorm1d(self.channels[2])
        self.bn4 = nn.BatchNorm1d(self.channels[1])
        self.bn5 = nn.BatchNorm1d(self.channels[0])
        self.fc1 = nn.Linear(self.channels[2], self.channels[1])
        self.fc2 = nn.Linear(self.channels[1], self.channels[0])
        self.fc3 = nn.Linear(self.channels[0], 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.channels[2])

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, device=x.device).view(1, 9).expand(batchsize, -1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, channels, k=64):
        super(STNkd, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv1d(k, self.channels[0], 1)
        self.conv2 = torch.nn.Conv1d(self.channels[0], self.channels[1], 1)
        self.conv3 = torch.nn.Conv1d(self.channels[1], self.channels[2], 1)
        self.fc1 = nn.Linear(self.channels[2], self.channels[1])
        self.fc2 = nn.Linear(self.channels[1], self.channels[0])
        self.fc3 = nn.Linear(self.channels[0], k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(self.channels[0])
        self.bn2 = nn.BatchNorm1d(self.channels[1])
        self.bn3 = nn.BatchNorm1d(self.channels[2])
        self.bn4 = nn.BatchNorm1d(self.channels[1])
        self.bn5 = nn.BatchNorm1d(self.channels[0])

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.channels[-1])

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).expand(batchsize, -1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
    
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.channels = [128, 256, 512]
        self.stn = STN3d(self.channels, channel)
        self.conv1 = torch.nn.Conv1d(channel, self.channels[0], 1)
        self.conv2 = torch.nn.Conv1d(self.channels[0], self.channels[1], 1)
        self.conv3 = torch.nn.Conv1d(self.channels[1], self.channels[2], 1)
        self.bn1 = nn.BatchNorm1d(self.channels[0])
        self.bn2 = nn.BatchNorm1d(self.channels[1])
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(self.channels, k=self.channels[0])

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = torch.max(x, 1, keepdim=True)[0]
        return x.transpose(1,2)