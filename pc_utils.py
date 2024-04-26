import math
import numbers
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from plyfile import PlyData
from pyntcloud import PyntCloud
from tqdm import tqdm
import matplotlib.pyplot as plt


def RandomScale(data, scales):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix
    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & s \\
        \end{bmatrix}
    for three-dimensional positions.
    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """
    assert isinstance(scales, (tuple, list)) and len(scales) == 2
    scale = random.uniform(*scales)
    data = data * scale
    return data


def RandomFlip(data, p=0.5, axis=-1):
    """Flips node positions along a given axis randomly with a given
    probability.
    Args:
        axis (int): The axis along the position of nodes being flipped.
        p (float, optional): Probability that node positions will be flipped.
            (default: :obj:`0.5`)
    """
    assert len(data.shape) == 2

    if axis == -1:
        dim1_flipped = False
        flip_dim = data.shape[0] if data.shape[1] > data.shape[0] else data.shape[1]
        for ax in range(0, flip_dim, 2):
            # y-axis flipping doesn't make sense, as occlusal is always y-axis
            p *= 0.5 if dim1_flipped else 1  # to lower likelihood of both flipped
            if random.random() < p:
                pos = data.clone()
                pos[..., ax] = -pos[..., ax]
                data = pos
                dim1_flipped = True
        return data
    else:
        if random.random() < p:
            pos = data.clone()
            pos[..., axis] = -pos[..., axis]
            data = pos
        return data


def RandomRotate(data, degrees, axis=-1):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.
    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """
    if isinstance(degrees, numbers.Number):
        degrees = (-abs(degrees), abs(degrees))
    assert isinstance(degrees, (tuple, list)) and len(degrees) == 2

    degree = math.pi * random.uniform(*degrees) / 180.0
    sin, cos = math.sin(degree), math.cos(degree)
    matrix = [[[1, 0, 0], [0, cos, sin], [0, -sin, cos]],  # axis 0
              [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]],  # axis 1
              [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]]  # axis 2
    if axis == -1:
        for ax in range(3):
            data = data @ torch.Tensor(matrix[ax]).t()
        return data
    else:
        data = data @ torch.Tensor(matrix[axis]).t()
        return data


def uniform_random_rotation(x):
    """Apply a random rotation in 3D, with a distribution uniform over the
    sphere.

    Arguments:
        x: vector or set of vectors with dimension (n, 3), where n is the
            number of vectors

    Returns:
        Array of shape (n, 3) containing the randomly rotated vectors of x,
        about the mean coordinate of x.

    Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
    https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    """

    def generate_random_z_axis_rotation():
        """Generate random rotation matrix about the z axis."""
        R = torch.eye(3)
        x1 = np.random.rand()
        R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
        R[0, 1] = -np.sin(2 * np.pi * x1)
        R[1, 0] = np.sin(2 * np.pi * x1)
        return R

    # There are two random variables in [0, 1) here (naming is same as paper)
    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()

    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation()
    v = torch.Tensor([
        np.cos(x2) * np.sqrt(x3),
        np.sin(x2) * np.sqrt(x3),
        np.sqrt(1 - x3)
    ])
    H = torch.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    x = x.reshape((-1, 3))
    mean_coord = torch.mean(x, dim=0)
    return ((x - mean_coord) @ M) + mean_coord @ M


def angle_axis_to_rotation_matrix(angle_axis):
    # Stolen from pytorch geometry
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix
    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.
    Returns:
        Tensor: tensor of 4x4 rotation matrices.
    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`
    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4


def slight_rotation(point_cloud, point_normals=None):
    """
    Applies a slight rotation and tilt to a given point cloud.

    Args:
        point_cloud (torch.Tensor): The input point cloud with shape (N, 3), where N is the number of points.
        point_normals (torch.Tensor, optional): The normals of the input point cloud with shape (N, 3). Defaults to None.

    Returns:
        tuple: A tuple containing the augmented point cloud and the augmented normals (if provided).
            - point_cloud_aug_homo (torch.Tensor): The augmented point cloud with shape (N, 3).
            - normals_aug (torch.Tensor): The augmented normals with shape (N, 3) if point_normals is not None, otherwise None.
    """
    # Generate random z axis rotation
    init_2_aug = torch.eye(4)
    y_axis_theta = (torch.rand(1) * np.pi / 5 - np.pi / 10)

    # y-axis rotation matrix
    R = torch.eye(3)
    R[0, 0] = R[2, 2] = torch.cos(y_axis_theta)
    R[0, 2] = torch.sin(y_axis_theta)
    R[2, 0] = -torch.sin(y_axis_theta)

    init_2_aug[:3, :3] = R

    # Do a slight tilt away from the y-axis
    theta = torch.rand(1) * 2. * np.pi
    tilt_axis = torch.Tensor([torch.cos(theta), 0, torch.sin(theta)])
    tilt_angle = torch.rand(1) * np.pi / 5 - np.pi / 10

    tilt_rotation = angle_axis_to_rotation_matrix(tilt_angle[None, ...] * tilt_axis[None, ...])[0]

    init_2_aug = init_2_aug @ tilt_rotation

    # Apply augmentation
    point_cloud_init_homo = torch.ones((point_cloud[:, :3].shape[0], 4))
    point_cloud_init_homo[:, :3] = point_cloud
    point_cloud_aug_homo = (init_2_aug @ point_cloud_init_homo.T).T
    point_cloud_aug_homo = point_cloud_aug_homo[:, :3] / point_cloud_aug_homo[:, -1, None]

    if point_normals is not None:
        normals_aug = (init_2_aug[:3, :3] @ point_normals.T).T
        return point_cloud_aug_homo, normals_aug
    else:
        return point_cloud_aug_homo, None


def calculate_point_cloud_std(folder, data_files):
    """ Calculate normalization values for new normalization"""
    value_sum, std_dumb_way, point_dist, scale_dict = 0, [], [], {}

    for idx, file in enumerate(tqdm(data_files)):
        ply_load = PlyData.read(file["ply_file"])

        point_cloud = np.array((ply_load.elements[0]['x'], ply_load.elements[0]['y'], ply_load.elements[0]['z']))
        point_dist.append(point_cloud.shape[1])

        scale_dict[file['id']] = np.max(np.linalg.norm(point_cloud, axis=0))

        std_dumb_way.append(np.linalg.norm(point_cloud, axis=0).std().item())

    std = np.mean(std_dumb_way)

    return std, point_dist, scale_dict


def sample_facets(idx, point_cloud, facet_areas, k, ply_load):
    """
    sampling from facets propertional to facet area size according to
    https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263
    :param idx: idx of specific point_cloud in dataaset
    :param point_cloud: specific point_cloud data [N, 3]
    :param facet_areas: list of facet area arrays of all point clouds in the dataset
    :param k: number of points to sample
    :param ply_load: Plydata loaded
    :return: pointcloud: the newly sampled point cloud
    """
    facet_idx = np.array((ply_load.elements[1]['v1'], ply_load.elements[1]['v2'], ply_load.elements[1]['v3']))
    facets = point_cloud[facet_idx.T, :]  # [num_points, point_in_facet, xyz]

    sampled_faces = random.choices(np.arange(facet_areas[idx].shape[0]), weights=facet_areas[idx],
                                   k=k)  # this is with replacement
    rand = torch.sort(torch.rand(k, 2), dim=1)[0]
    s, t = rand[:, 0], rand[:, 1]
    point_cloud = s.unsqueeze(1) * facets[sampled_faces, 0, :] + (t - s).unsqueeze(1) * facets[sampled_faces, 1,
                                                                                        :] + (1 - t).unsqueeze(
        1) * facets[sampled_faces, 2, :]
    return point_cloud


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.backprop_num = 0

    def batch_pairwise_dist_new(self, x, y):
        xx = x.pow(2).sum(dim=2)
        yy = y.pow(2).sum(dim=2)
        zz = torch.bmm(x, y.transpose(2, 1))
        # just repeating of x_1^2, x_2^2 ... x_n^2
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)  # x^2 + y^2 - 2xy
        return P

    def forward(self, ground_truth, reconstruction, surface_normals_tuple=None):

        #if surface_normals_tuple is None
        P = self.batch_pairwise_dist_new(ground_truth, reconstruction)
        mins, min_idx = torch.min(P, 1)  # from pred to gts
        loss_1 = torch.mean(torch.clamp(mins, min=1e-10))
        mins, _ = torch.min(P, 2)  # from gts to pred
        loss_2 = torch.mean(torch.clamp(mins, min=1e-10))
        return {"total_loss":(loss_1 + loss_2) * 1000,
                "chamfer": (loss_1 + loss_2) * 1000,}
    


def save_pointcloud(point_cloud, save_path, point_normals=None):
    if point_cloud.shape[2] == 3:
        point_cloud = point_cloud.transpose((0, 2, 1))
    if point_normals is None:
        d = {'x': point_cloud[0, 0, :], 'y': point_cloud[0, 1, :],'z': point_cloud[0, 2, :]}
    else:
        if point_normals.shape[2] == 3:
            point_normals = point_normals.transpose((0, 2, 1))
        d = {'x': point_cloud[0, 0, :], 'y': point_cloud[0, 1, :], 'z': point_cloud[0, 2, :],
             'nx': point_normals[0, 0, :], 'ny': point_normals[0, 1, :], 'nz': point_normals[0, 2, :]}
    cloud = PyntCloud(pd.DataFrame(data=d))
    cloud.to_file(save_path)


def save_ply_manual(verts, filename, faces):
    if verts.shape[1] != 3:
        verts = verts.transpose(0,1)

    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % verts.shape[0])
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('element face %d\n' % len(faces))
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        for i in range(verts.shape[0]):
            f.write('%f %f %f\n' % (verts[i, 0], verts[i, 1], verts[i, 2]))
        for i in range(len(faces)):
            f.write('3 %d %d %d\n' % (faces[i, 0], faces[i, 1], faces[i, 2]))


def facets_from_grid(num_points, reverse_facets=False):
    """ num_points should be points in sqrt(n_points_in_grid) """
    facet_list = []
    # start from 1 to go one down
    for i in range(1, num_points * num_points - num_points):
        if i % num_points != 0:
            if not reverse_facets:
                facet_list.append(np.array([i - 1, i, i + num_points - 1]))
                facet_list.append(np.array([i, i + num_points, i + num_points- 1]))
            else:
                facet_list.append(np.array([i - 1, i + num_points - 1, i]))
                facet_list.append(np.array([i, i + num_points-1, i + num_points]))
    return np.array(facet_list)

def save_cloud_rgb(cloud, red, green, blue, filename):
    cloud = cloud.cpu()
    d = {'x': cloud[0],
         'y': cloud[1],
         'z': cloud[2],
         'red': red,
         'green': green,
         'blue': blue}
    cloud_pd = pd.DataFrame(data=d)
    cloud_pd[['red', 'green', 'blue']] = cloud_pd[['red', 'green', 'blue']].astype(np.uint8) 
    cloud = PyntCloud(cloud_pd)
    cloud.to_file(filename)
 