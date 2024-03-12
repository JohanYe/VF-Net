import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import knn, local_maxpool, local_cov, Residual_Linear_Layer


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
