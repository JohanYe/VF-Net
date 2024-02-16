import torch
import torch.nn as nn
import torch.nn.functional as F

from model.foldingnet.foldingnet_utils import knn, local_maxpool, local_cov, Residual_Linear_Layer


class FoldNet_Encoder_Linear(nn.Module):
    def __init__(self, args):
        super(FoldNet_Encoder_Linear, self).__init__()
        if args.k == None:
            self.k = 16
        else:
            self.k = args.k
        self.n = 2048  # input point cloud size
        self.point_normals = args.point_normals
        self.in_channels = 15 if args.point_normals else 12
        self.mlp1 = nn.Sequential(
            nn.Linear(self.in_channels, 128),
            nn.ReLU(),
            Residual_Linear_Layer(128, 128),
            nn.ReLU(),
            Residual_Linear_Layer(128, 128),
            nn.ReLU(),
        )
        self.linear1 = nn.Sequential(Residual_Linear_Layer(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 256),
                                     nn.ReLU())

        self.linear2 = nn.Sequential(Residual_Linear_Layer(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 512))
        self.mlp2 = nn.Sequential(
            nn.Linear(512, int(2*args.feat_dims)),
            nn.ReLU(),
            nn.Linear(int(2*args.feat_dims), int(2*args.feat_dims))
        )

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)  # (batch_size, num_points, 64)
        feat2 = self.linear1(x)
        x = local_maxpool(feat2, idx)  # (batch_size, num_points, 128)
        x = self.linear2(x)  # (batch_size, num_points, 256)
        return x, feat2

    def forward(self, pts):
        pts = pts.transpose(2, 1)  # (batch_size, 3, num_points)
        if self.point_normals:
            pts_in = pts[:, :3, :]
            pns_in = pts[:, 3:, :]
        else:
            pts_in = pts
        idx = knn(pts_in, k=self.k)
        out = local_cov(pts_in, idx)  # (batch_size, 3, num_points) -> (batch_size, 12, num_points])
        x = torch.cat([out, pns_in.transpose(1, 2)], dim=-1) if self.point_normals else out
        feat1 = self.mlp1(x)  # (batch_size, num_points, 12) -> (batch_size, num_points, 64])
        feat3, feat2 = self.graph_layer(feat1, idx)  # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        global_cat = torch.max(feat3, 1, keepdim=True)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1, 1024)
        # global_cat = torch.cat((feat1, feat2, feat3, global_feat.repeat(1, x.shape[1], 1)), dim=-1)
        feat = self.mlp2(global_cat)
        feat = torch.max(feat, 1, keepdim=True)[0]
        return feat, feat1  # (batch_size, 1, feat_dims), (batch_size, num_points, 64)


class FoldNet_Encoder_Conv(nn.Module):
    def __init__(self, args):
        super(FoldNet_Encoder_Conv, self).__init__()
        if args.k == None:
            self.k = 16
        else:
            self.k = args.k
        self.n = 2048  # input point cloud size
        self.point_normals = args.point_normals
        self.in_channels = 15 if args.point_normals else 12
        self.mlp1 = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 256)
        self.conv2 = nn.Conv1d(256, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        )

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)
        x = self.linear1(x).transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x.transpose(1, 2), idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    # TODO: add point normals
    # TODO: change transposes so that they are needed in linear version instead of conv version
    def forward(self, pts):
        if pts.shape[-1] == 3:
            pts = pts.transpose(2, 1)  # (batch_size, 3, num_points)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx).transpose(1, 2)  # (batch_size, 3, num_points) -> (batch_size, 12, num_points])
        feat1 = self.mlp1(x).transpose(1,2)  # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        x = self.graph_layer(feat1, idx)  # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = self.mlp2(x)  # (batch_size, 1024, 1) -> (batch_size, feat_dims, 1)
        feat = x.transpose(2, 1)  # (batch_size, feat_dims, 1) -> (batch_size, 1, feat_dims)
        return feat, feat1.transpose(1, 2)  # (batch_size, 1, feat_dims), (batch_size, num_points, 64)
