import json
import os
import platform
import random
from glob import glob
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

import h5py
import numpy as np
import torch
import torch.utils.data as data
from plyfile import PlyData
from torch.utils.data import Dataset
from tqdm import tqdm

import pc_utils



def few_unns(data_dir, unn_list):
    case_dir = os.listdir(data_dir)
    data_info = []
    for dir in case_dir:
        d = {}
        for unn_n in unn_list:
            if os.path.isfile(os.path.join(data_dir, dir, f"unn{unn_n}.ply")):
                d["id"] = f"{dir}_{unn_n}"
                d["ply_file"] = os.path.join(data_dir, dir, f"unn{unn_n}.ply")
                data_info.append(d)
    return data_info

def published_data(data_dir):
    case_dir = os.listdir(data_dir)
    case_dir = [s for s in case_dir if "_point_cloud.ply" in s]
    data_info = []
    for file in case_dir:
        d = {}
        file_name_split = file.split("_")
        d["id"] = f"{file_name_split[0]}_3"
        d["ply_file"] = os.path.join(data_dir, file)
        data_info.append(d)
    return data_info


def _parse_path(path):
    if not path.startswith("file"):
        return path
    parsed = urlparse(path)
    host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
    return os.path.abspath(os.path.join(host, url2pathname(unquote(parsed.path))))


def all_unn(data_dir):
    """
    consider just making this all teeth
    """
    import xml.etree.ElementTree as ET

    xml_file = [x for x in os.listdir(data_dir) if x[-4:] == ".xml"]
    assert len(xml_file) <= 2
    tree = ET.parse(os.path.join(data_dir, xml_file[0]))
    root = tree.getroot()
    data_info = []
    for e in root.iter("DataItem"):
        d = {}
        non_primary_tooth = False
        for data_element in e.iter("Tag"):
            if data_element.attrib["key"] == "unn":
                if int(data_element.text) < 33:
                    non_primary_tooth = True
                d["id"] = f'{e.attrib["Id"]}_{data_element.text}'
            elif "sh" in xml_file[0]: # sorting hat data only
                if data_element.attrib["key"] == "AnnotatorName":
                    continue
                d[data_element.attrib["key"]] = data_element.text == "True"
            
            mesh_element = e.find("*/[@name='mesh']")
            d["ply_file"] = _parse_path(mesh_element.text)
            
        if non_primary_tooth:  # exclude a small amount of teeth with labels over 32 (4k out of 130k)
            data_info.append(d)
    return data_info


def tooth_wear_only(data_dir):
    import xml.etree.ElementTree as ET

    xml_file = [x for x in os.listdir(data_dir) if x[-4:] == ".xml"]
    assert len(xml_file) == 1
    tree = ET.parse(os.path.join(data_dir, xml_file[0]))
    root = tree.getroot()
    data_info = []
    for e in root.iter("DataItem"):
        d = {}
        non_primary_tooth = False
        for data_element in e.iter("Tag"):
            if data_element.attrib["key"] == "unn":
                if int(data_element.text) < 33:
                    non_primary_tooth = True
                d["id"] = f'{e.attrib["Id"]}_{data_element.text}'
            elif "sh" in xml_file[0]: # sorting hat data only
                if data_element.attrib["key"] == "AnnotatorName":
                    continue
                d[data_element.attrib["key"]] = data_element.text == "True"
            
            mesh_element = e.find("*/[@name='mesh']")
            d["ply_file"] = _parse_path(mesh_element.text)
            
        if non_primary_tooth:  # exclude a small amount of teeth with labels over 32 (4k out of 130k)
            data_info.append(d)
    return data_info


class Teeth_Dataset(Dataset):
    def __init__(self, unn, folder_path, is_train, args, global_pc_std=None, cluster=False, only_worn_teeth=False, k=2048):
        self.is_train = is_train
        self.data_folder_path = folder_path
        self.cluster = cluster
        self.k = k
        self.point_normals = args.point_normals
        self.global_pc_std = global_pc_std
        # self.all_points_mean = torch.Tensor([-0.0266, 0.3486, -0.0049]).unsqueeze(1)
        # self.all_points_mean = torch.Tensor([-0.05251455, 0.48561925, -0.05256953]).unsqueeze(1)

        if unn == "all":
            self.data = all_unn(folder_path)
        elif unn == ["3"]:
            self.data = published_data(folder_path)
        else:
            self.data = few_unns(folder_path, unn_list=unn)

        if only_worn_teeth:
            self.data = [d for d in self.data if not d["Hole"] and
                                                 not d["Filling"] and
                                                 not d["Restoration"] and
                                                 not d["Severe problems"] and
                                                 not d["Spit bubble"] and
                                                 not d["Bracket"]]

        # if args.model.lower() == "vae":
        # if self.is_train:
        self.data = self.cull_data(self.data, self.k)

        if self.global_pc_std is None:
            print("Calculating global point cloud std for normalization")
            self.global_pc_std, self.point_dist, self.scale_dict = pc_utils.calculate_point_cloud_std(folder_path, self.data)

        if self.is_train:
            scale = (torch.ones(3)/2) / self.global_pc_std  # remember to adjust for normalization
            self.translate = torch.distributions.normal.Normal(loc=torch.Tensor([0, 0, 0]),
                                                               scale=scale)
            self.random_noise = torch.distributions.normal.Normal(loc=0, scale=0.0005)


    def __len__(self):
        return len(self.data)

    def cull_data(self, data, min_pc_size):
        """
        Used to remove point clouds below threshold size from data.
        Primarily useful for VAE cases where it is important for loss calculation
        :param data: list of dicts with datapath and id
        :param min_pc_size: threshold size
        :return: new list of dicts with datapath and id
        """
        new_data_dict = []
        print("Removing samples not meeting max_pc_n for VAE")
        for data_dict in tqdm(data):
            ply_load = PlyData.read(data_dict['ply_file'])
            point_cloud = torch.from_numpy(
                np.array((ply_load.elements[0]['x'], ply_load.elements[0]['y'], ply_load.elements[0]['z'])))
            
            # check if still large enough after edge removal
            with open(data_dict['ply_file'][:-15] + "extraInfo.dat", "rb") as f:
                extra_info = np.fromfile(f, np.double)
            extra_info = extra_info.reshape(-1, point_cloud.shape[-1]).T  # (N_points, 2) curvature and dist from border
            distance_from_border = extra_info[:, 1]
            delete_indices = np.where(distance_from_border < 0.5)[0]  # this value could be investigated furter
            if (point_cloud.shape[1] - len(delete_indices)) > min_pc_size:
                data_dict['data'] = point_cloud
                new_data_dict.append(data_dict)

            # mass midpoint metric test
            # indices = torch.randperm(point_cloud.shape[1])[:self.k]
            # y_boundary = point_cloud.max(1)[0] - point_cloud.min(1)[0]
            # point_cloud = point_cloud[:, indices]
            # mass_midpoint = point_cloud.mean(dim=1)
            # x = point_cloud[:,point_cloud[1,:] > (mass_midpoint[1] - 0.1*y_boundary[1])]
            # pc_utils.save_pointcloud(x.unsqueeze(0).detach().cpu().numpy(),f"./test/{data_dict['id']}.ply")
        return new_data_dict

    def pc_normalize(self, pc):
        """ dunno if good idea, don't blame me if used """
        # pc = (pc - self.all_points_mean) / self.global_pc_std  # value comes from std(norms(fc_train_3))
        pc = pc / self.global_pc_std
        return pc

    def pc_unnormalize(self, pc):

        # pc = (pc + self.all_points_mean) * self.global_pc_std  # value comes from std(norms(fc_train_3))
        pc = pc*self.global_pc_std
        return pc
    
    def get_specific_sample(self, id):
        for i, d in enumerate(self.data):
            if d["id"] == id:
                return self.__getitem__(i)
            
        # if not found
        print("id not found")
        return None          

    def __getitem__(self, item):
        ply_path = self.data[item]['ply_file']
        ply_load = PlyData.read(ply_path)
        if 'property list uchar int vertex_indices' in ply_load.header:
            for f in self.data:
                print("preprocessing files")
                pc_utils.preprocessing(os.path.join(self.folder_path, f))
            ply_load = PlyData.read(os.path.join(ply_path))
        point_cloud = torch.from_numpy(np.array((ply_load.elements[0]['x'], ply_load.elements[0]['y'], ply_load.elements[0]['z'])))
        point_cloud = self.pc_normalize(point_cloud).transpose(0, 1)

        with open(ply_path[:-15] + "extraInfo.dat", "rb") as f:
            extra_info = np.fromfile(f, np.double)
        # curvature = torch.from_numpy(curvature[indices])
        extra_info = extra_info.reshape(-1, point_cloud.shape[0]).T  # (N_points, 2) curvature and dist from border
        distance_from_border = extra_info[:, 1]

        delete_indices = np.where(distance_from_border < 0.5)[0]  # this value could be investigated furter
        delete_points_near_border = np.delete(np.arange(point_cloud.shape[0]), delete_indices)
        n_samples = min(self.k, len(delete_points_near_border))
        indices = np.random.choice(delete_points_near_border, n_samples, replace=False)
        # if n_samples < self.k:
        #     missing_points = np.random.choice(delete_indices, self.k - n_samples, replace=False)
        #     indices = np.concatenate((indices, missing_points))
        point_cloud = point_cloud[indices]
        
        if self.point_normals:
            point_normals = torch.from_numpy(
                np.array((ply_load.elements[0]['nx'], ply_load.elements[0]['ny'], ply_load.elements[0]['nz'])))
            point_normals = point_normals.transpose(0, 1)[indices]
        else:
            point_normals = None

        if self.is_train and np.random.random() > 0.5:  # Augmentation
            # point_cloud = pc_utils.RandomFlip(point_cloud, p=0.33, axis=0)  # only flip x due to library coordinates
            point_cloud = pc_utils.RandomScale(point_cloud, scales=[0.8, 1.2])
            # point_cloud = pc_utils.RandomRotate(point_cloud, 180)
            point_cloud, point_normals = pc_utils.slight_rotation(point_cloud, point_normals)
            # point_cloud = pc_utils.uniform_random_rotation(point_cloud)
            # point_cloud += self.translate.sample().unsqueeze(0)
            # point_cloud += self.random_noise.sample(point_cloud.size())

        point_cloud = torch.cat((point_cloud, point_normals), dim=1) if self.point_normals else point_cloud

        return dict(pc=point_cloud, ids=self.data[item]['id']) #.split("_")[-1] + "_" + self.data[item]['id'].split("_")[1])

class Teeth_Dataset_Unlimited_Points(Teeth_Dataset):
    def __init__(self, unn, folder_path, is_train, args, global_pc_std=None, cluster=False, only_worn_teeth=False):
        super().__init__(unn, folder_path, is_train, args, global_pc_std, cluster, only_worn_teeth)
        self.k = 1000000
        self.data = self.cull_data(self.data, 2048)


class Baseline_Dataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40',
                 num_points=2048, split='train', load_name=False,
                 random_rotate=False, random_jitter=True, random_translate=False):

        assert dataset_name.lower() in ['shapenetcorev2',
                                        'shapenetpart', 'modelnet10', 'modelnet40']
        assert num_points <= 2048

        if dataset_name in ['shapenetpart', 'shapenetcorev2']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        self.root = os.path.join(root, dataset_name + '*hdf5_2048')
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        self.k = num_points

        self.path_h5py_all = []
        self.path_json_all = []
        if self.split in ['train', 'trainval', 'all']:
            self.get_path('train')
        if self.dataset_name in ['shapenetpart', 'shapenetcorev2']:
            if self.split in ['val', 'trainval', 'all']:
                self.get_path('val')
        if self.split in ['test', 'all']:
            self.get_path('test')

        self.path_h5py_all.sort()
        data, label = self.load_h5py(self.path_h5py_all)
        if self.load_name:
            self.path_json_all.sort()
            self.name_all = self.load_json(self.path_json_all)  # load label name
            self.class_matches = np.array(self.name_all) == "airplane"  # choose your own class
            self.name = np.array(self.name_all)[self.class_matches].tolist()
            self.data = np.concatenate(data, axis=0)[self.class_matches]
            self.label = np.concatenate(label, axis=0)[self.class_matches]
        else:
            self.data = np.concatenate(data, axis=0)
            self.label = np.concatenate(label, axis=0)

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5' % type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json' % type)
            self.path_json_all += glob(path_json)
        return

    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        return all_data, all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j = open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)

        if self.load_name:
            return {"pc":point_set, "ids":label, "name": name}
        else:
            return {"pc":point_set, "ids":label}

    def __len__(self):
        return self.data.shape[0]


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


class Uniform15KPC(torch.utils.data.Dataset):
    def __init__(self, root, subdirs, tr_sample_size=10000, te_sample_size=10000, split='train', scale=1.,
                 standardize_per_shape=False,
                 normalize_per_shape=False, random_offset=False, random_subsample=False, normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None, input_dim=3):
        self.root = root
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_offset = random_offset
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        if split == 'train':
            self.max = tr_sample_size
        elif split == 'val':
            self.max = te_sample_size
        else:
            self.max = max((tr_sample_size, te_sample_size))

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
            for mid in all_mids:
                # obj_fname = os.path.join(sub_path, x)
                obj_fname = os.path.join(root, subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)
                except:
                    continue

                assert point_cloud.shape[0] == 15000
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.standardize_per_shape = standardize_per_shape
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:  # per shape normalization
            raise NotImplementedError("normalize_per_shape==True is deprecated")
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        # self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        # JY: added, due to folding wanting the data in a specific area
        self.all_points = (self.all_points - self.all_points_mean)
        self.all_points_std = np.array([[[1]]])
        self.global_pc_std = np.array([[[1]]])


        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d" % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def save_statistics(self, save_dir):
        np.save(os.path.join(save_dir, f"{self.split}_set_mean.npy"), self.all_points_mean)
        np.save(os.path.join(save_dir, f"{self.split}_set_std.npy"), self.all_points_std)
        np.save(os.path.join(save_dir, f"{self.split}_set_idx.npy"), np.array(self.shuffle_idx))

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        tr_ofs = tr_out.mean(0, keepdim=True)
        te_ofs = te_out.mean(0, keepdim=True)

        if self.standardize_per_shape:
            # If standardize_per_shape, centering in/out
            tr_out -= tr_ofs
            te_out -= te_ofs
        if self.random_offset:
            # scale data offset
            if random.uniform(0., 1.) < 0.2:
                scale = random.uniform(1., 1.5)
                tr_out -= tr_ofs
                te_out -= te_ofs
                tr_ofs *= scale
                te_ofs *= scale
                tr_out += tr_ofs
                te_out += te_ofs

        m, s = self.get_pc_stats(idx)
        m, s = torch.from_numpy(np.asarray(m)), torch.from_numpy(np.asarray(s))
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        return {
            'ids': idx,
            'set': tr_out if self.split == 'train' else te_out,
            'offset': tr_ofs if self.split == 'train' else te_ofs,
            'mean': m, 'std': s, 'label': cate_idx,
            'sid': sid, 'mid': mid
        }


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, root="/data/shapenet/ShapeNetCore.v2.PC15k",
                 categories=['airplane'], tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 standardize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_offset=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        self.k = tr_sample_size
        self.root = root
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        assert 'v2' in root, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super().__init__(
            root, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split,
            scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            standardize_per_shape=standardize_per_shape,
            random_offset=random_offset,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean,
            all_points_std=all_points_std,
            input_dim=3)

        print(f"Done!")


def collate_fn(batch):
    ret = dict()
    for k, v in batch[0].items():
        ret.update({k: [b[k] for b in batch]})

    mean = torch.stack(ret['mean'], dim=0)  # [B, 1, 3]
    std = torch.stack(ret['std'], dim=0)  # [B, 1, 1]

    s = torch.stack(ret['set'], dim=0)  # [B, N, 3]
    offset = torch.stack(ret['offset'], dim=0)
    mask = torch.zeros(s.size(0), s.size(1)).bool()  # [B, N]
    cardinality = torch.ones(s.size(0)) * s.size(1)  # [B,]

    ret.update({'pc': s, 'offset': offset, 'set_mask': mask, 'cardinality': cardinality,
                'mean': mean, 'std': std})
    return ret


def build(args):
    train_dataset = ShapeNet15kPointClouds(
        categories=["airplane"],
        split='train',
        tr_sample_size=2048, #args.tr_max_sample_points,
        te_sample_size=2048, #args.te_max_sample_points,
        scale=1., #args.dataset_scale,
        root="/train/ShapeNetCore.v2.PC15k/",#args.shapenet_data_dir,
        standardize_per_shape=False, #args.standardize_per_shape,
        normalize_per_shape=False, #args.normalize_per_shape,
        normalize_std_per_axis=False, #args.normalize_std_per_axis,
        random_subsample=True)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn,
        worker_init_fn=init_np_seed)

    val_dataset = ShapeNet15kPointClouds(
        categories=["airplane"],
        split='val',
        tr_sample_size=2048, #args.tr_max_sample_points,
        te_sample_size=2048, #args.te_max_sample_points,
        scale=1., #args.dataset_scale,
        root="/train/ShapeNetCore.v2.PC15k",#args.shapenet_data_dir,
        standardize_per_shape=False, #args.standardize_per_shape,
        normalize_per_shape=False, #args.normalize_per_shape,
        normalize_std_per_axis=False, #args.normalize_std_per_axis,
        all_points_mean=train_dataset.all_points_mean,
        all_points_std=train_dataset.all_points_std)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn,
        worker_init_fn=init_np_seed)

    data_loaders = {"Train": train_loader, "Test": val_loader}

    return train_dataset, val_dataset, data_loaders


class Teeth_Dataset_PointFlow(Dataset):
    """
    Dataloader for point flow repository
    """
    def __init__(self, is_train=True, unn="all", global_pc_std=4.8647394309783545, cluster=False):
        self.is_train = is_train
        base_path = "/train/pcvae/pcvae_v1.3_"
        self.data_folder_path = base_path + "train/" if self.is_train else base_path + "test/"
        self.cluster = cluster
        self.k = 100000
        self.point_normals = False
        self.global_pc_std = global_pc_std

        if unn == "all":
            self.data = all_unn(self.data_folder_path)
        else:
            self.data = few_unns(self.data_folder_path, unn_list=unn)

        # if args.model.lower() == "vae":
        #    self.data = self.cull_data(self.data, args.max_pc_n)
        # self.data = self.cull_data(self.data, 1000)  # temporary for experiment

        if self.global_pc_std is None:
            print("Calculating global point cloud std for normalization")
            self.global_pc_std, self.point_dist, self.scale_dict = pc_utils.calculate_point_cloud_std(folder_path,
                                                                                                      self.data)

        if self.is_train:
            scale = (torch.ones(3) / 2) / self.global_pc_std  # remember to adjust for normalization
            self.translate = torch.distributions.normal.Normal(loc=torch.Tensor([0, 0, 0]),
                                                               scale=scale)
            self.random_noise = torch.distributions.normal.Normal(loc=0, scale=0.002)

    def __len__(self):
        return len(self.data)

    def cull_data(self, data, min_pc_size):
        """
        Used to remove point clouds below threshold size from data.
        Primarily useful for VAE cases where it is important for loss calculation
        :param data: list of dicts with datapath and id
        :param min_pc_size: threshold size
        :return: new list of dicts with datapath and id
        """
        new_data_dict = []
        print("Removing samples not meeting max_pc_n for VAE")
        for data_dict in tqdm(data):
            ply_load = PlyData.read(data_dict['ply_file'])
            point_cloud = torch.from_numpy(
                np.array((ply_load.elements[0]['x'], ply_load.elements[0]['y'], ply_load.elements[0]['z'])))
            if point_cloud.shape[1] > min_pc_size:
                new_data_dict.append(data_dict)
        return new_data_dict

    def pc_normalize(self, pc):
        """ dunno if good idea, don't blame me if used """
        pc = pc / self.global_pc_std  # value comes from std(norms(fc_train_3))
        return pc

    def pc_unnormalize(self, pc):
        """ dunno if good idea, don't blame me if used """

        # actual operation
        pc = pc * self.global_pc_std  # value comes from std(norms(fc_train_3))
        return pc

    def __getitem__(self, item):
        ply_path = self.data[item]['ply_file']
        ply_load = PlyData.read(ply_path)
        if 'property list uchar int vertex_indices' in ply_load.header:
            for f in self.data:
                print("preprocessing files")
                pc_utils.preprocessing(os.path.join(self.folder_path, f))
            ply_load = PlyData.read(os.path.join(ply_path))
        point_cloud = torch.from_numpy(
            np.array((ply_load.elements[0]['x'], ply_load.elements[0]['y'], ply_load.elements[0]['z'])))
        point_cloud = self.pc_normalize(point_cloud).transpose(0, 1)

        indices = torch.randperm(point_cloud.shape[0])[:self.k]
        point_cloud = point_cloud[indices]

        if self.point_normals:
            point_normals = torch.from_numpy(
                np.array((ply_load.elements[0]['nx'], ply_load.elements[0]['ny'], ply_load.elements[0]['nz'])))
            point_normals = point_normals.transpose(0, 1)[indices]

        if self.is_train and np.random.random() > 0.5:  # Augmentation
            point_cloud = pc_utils.RandomFlip(point_cloud, p=0.33, axis=0)  # only flip x due to library coordinates
            point_cloud = pc_utils.RandomScale(point_cloud, scales=[0.8, 1.2])
            # point_cloud = pc_utils.RandomRotate(point_cloud, 180)
            point_cloud = pc_utils.slight_rotation(point_cloud)
            # point_cloud = pc_utils.uniform_random_rotation(point_cloud)
            # point_cloud += self.translate.sample().unsqueeze(0)
            point_cloud += self.random_noise.sample(point_cloud.size())

        point_cloud = torch.cat((point_cloud, point_normals), dim=1) if self.point_normals else point_cloud

        return {
            'idx': item,
            'train_points': point_cloud,
            'test_points': point_cloud,
            'mean': torch.zeros(3), 'std': torch.Tensor(self.global_pc_std), 'cate_idx': self.data[item]['id'].split("_")[-1],
            # 'sid': sid, 'mid': mid
        }
