import argparse
import itertools
import json
import os
import shutil
import sys

import numpy as np
from sklearn.decomposition import PCA
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader
import utils
from model import vfnet
import pc_utils
from nflows import transforms, distributions, flows
import torch.nn as nn
from model.model_utils import Residual_Linear_Layer
import metrics



parser = argparse.ArgumentParser(description='Training loop boiler plate')
parser.add_argument('--dataset', required=False, default="teeth",
                    help="Data set, pick from ['teeth', 'modelnet40', 'shapenetcore']")
parser.add_argument('--x_train', required=False, help='Path to X_train folder', dest="x_train")
parser.add_argument('--x_val', required=False, help='Path to X_val folder', dest="x_val")
parser.add_argument('--x_test', required=False, help='Path to X_test folder', dest="x_test")
parser.add_argument('--model_path', required=False, default="", help='Path to state dict to continue training from')
parser.add_argument('--flow_num_epochs', required=False, type=int, default=150)
parser.add_argument('--num_workers', required=False, type=int, default=0)
parser.add_argument('--pe_num_epochs', required=False, type=int, default=300)
parser.add_argument('--seed', required=False, type=int, required=True)
parser.add_argument('--flow_viz_epochs', required=False, type=int, default=20)
parser.add_argument("--flow_n_layers", required=False, type=int, default=5)
parser.add_argument("--one_fold", action="store_true", required=False, default=False)
parser.add_argument("--test_name", required=False, type=str, default=None,
                    help="Name for test")

class OneFold(nn.Module):
    def __init__(self, feat_dims=256):
        super().__init__()
        self.OneFold = nn.Sequential(
            nn.Linear(514, feat_dims),
            nn.Dropout(0.2),
            nn.ReLU(),
            Residual_Linear_Layer(feat_dims, feat_dims),
            nn.ReLU(),
            Residual_Linear_Layer(feat_dims, feat_dims),
            nn.ReLU(),
            nn.Linear(feat_dims, 2),
            nn.Tanh()
        )
        self.num_points_sqrt = int(np.sqrt(2048))
        self.meshgrid = [[-1, 1, self.num_points_sqrt], [-1, 1, self.num_points_sqrt]]
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        self.meshgrid = np.array(list(itertools.product(x, y)))
    
    def forward(self, x, add_n_points=0):
        if add_n_points > 0:
            added_points = np.random.uniform(-1, 1, (add_n_points, 2))
            grid = torch.from_numpy(np.concatenate([self.meshgrid, added_points], axis=0)).float().to(x.device)
        else:
            grid = torch.from_numpy(self.meshgrid).float().to(x.device)
        x = x.repeat(1, grid.shape[0], 1)

        x = torch.cat([x, grid.repeat(x.shape[0],1,1)], dim=-1)
        x = self.OneFold(x)
        return x


class TwoFold(OneFold):
    def __init__(self, feat_dims=256):
        super().__init__(feat_dims)
        self.TwoFold = nn.Sequential(
            nn.Linear(514, feat_dims),
            nn.Dropout(0.2),
            nn.ReLU(),
            Residual_Linear_Layer(feat_dims, feat_dims),
            nn.ReLU(),
            Residual_Linear_Layer(feat_dims, feat_dims),
            nn.ReLU(),
            nn.Linear(feat_dims, feat_dims),
            nn.ReLU(),
            nn.Linear(feat_dims, 2),
            nn.Tanh()
        )
    
    def forward(self, latent_code, add_n_points=0):
        if add_n_points > 0:
            added_points = np.random.uniform(-1, 1, (add_n_points, 2))
            grid = torch.from_numpy(np.concatenate([self.meshgrid, added_points], axis=0)).float().to(latent_code.device)
        else:
            grid = torch.from_numpy(self.meshgrid).float().to(latent_code.device)
        latent_code = latent_code.repeat(1, grid.shape[0], 1)

        x = torch.cat([latent_code, grid.repeat(latent_code.shape[0],1,1)], dim=-1)
        x = self.OneFold(x)
        x = torch.cat([x, latent_code], dim=-1)
        x = self.TwoFold(x)
        return x

def collate_fn(batch):
    min_points = np.min([b["pc"].shape[0] for b in batch])
    diff_from_max = [min_points - b["pc"].shape[0] for b in batch]
    pc = []

    for idx, diff in enumerate(diff_from_max):
        if diff == 0:
            pc.append(batch[idx]["pc"])
        else:
            indices = torch.randperm(batch[idx]["pc"].shape[0])[:min_points]
            pc.append(batch[idx]["pc"][indices])

    pc = torch.stack(pc).transpose(1, 2)
    ids = [b["ids"] for b in batch]
    return {"pc": pc, "ids": ids}

def batch_pairwise_dist_new(x, y):
    xx = x.pow(2).sum(dim=2)
    yy = y.pow(2).sum(dim=2)
    zz = torch.bmm(x, y.transpose(2, 1))
    # just repeating of x_1^2, x_2^2 ... x_n^2
    rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy.unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)  # x^2 + y^2 - 2xy
    return P


def get_latents(args, model_dir):
    unn_list = args.unn.split(",") if args.unn != "all" else "all"
    teeth_std = 9.8186 if unn_list == ['3'] else 11.75121  
    train_set = dataloader.Teeth_Dataset(unn=unn_list,
                                         folder_path=args.x_train,
                                         is_train=False,
                                         global_pc_std=teeth_std,
                                         # global std(norm(pc)) from fc_train_unn_3
                                         args=args)
    val_set = dataloader.Teeth_Dataset(unn=unn_list,
                                        folder_path=args.x_val,
                                        is_train=False,
                                        global_pc_std=teeth_std,  # global std(norm(pc)) from fc_train_unn_3
                                        args=args)
    test_set = dataloader.Teeth_Dataset(unn=unn_list,
                                        folder_path=args.x_test,
                                        is_train=False,
                                        global_pc_std=teeth_std,  # global std(norm(pc)) from fc_train_unn_3
                                        args=args)

    data_loaders = {
        "Train": DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.num_workers,
                            pin_memory=False, collate_fn=collate_fn, drop_last=True),
        "Validation": DataLoader(val_set, batch_size=1, shuffle=True, num_workers=args.num_workers,
                            pin_memory=False, collate_fn=collate_fn, drop_last=True),
        "Test": DataLoader(test_set, batch_size=1, shuffle=True, num_workers=args.num_workers,
                           pin_memory=False, collate_fn=collate_fn, drop_last=True)}

    # model stuff
    vae = vfnet.Variational_autoencoder(args, num_points=train_set.k).to(args.device)
    vae = utils.load_pretrained_model(vae, load_path=args.model_path)
    vae.eval()

    device = args.device
    train_mu, train_std, val_mu, val_std, test_mu, test_std = [], [], [], [], [], []
    train_pe, val_pe, test_pe = [], [], []

    for phase in ["Train", "Validation", "Test"]:
        for idx, d in enumerate(tqdm(data_loaders[phase])):
            with torch.no_grad():
                pc_X, jaw_id = d["pc"], d["ids"]

                pc_X = pc_X.to(device)
                feature, feat1 = vae.encoder(pc_X.transpose(1,2))
                mu, lv = torch.chunk(feature, 2, dim=-1)
                z = vae.reparameterize(feature)
                z_repeated = z.repeat(1, feat1.shape[1], 1)
                cat = torch.cat((z_repeated, feat1), dim=-1)
                grid = vae.decoder.grid_map(cat) # (batch_size, num_points, 2)

                if phase == "Train":
                    train_mu.append(mu.cpu().numpy())
                    train_std.append(lv.cpu().numpy())
                    train_pe.append(grid.cpu().numpy())
                elif phase == "Validation":
                    val_mu.append(mu.cpu().numpy())
                    val_std.append(lv.cpu().numpy())
                    val_pe.append(grid.cpu().numpy())
                else:
                    test_mu.append(mu.cpu().numpy())
                    test_std.append(lv.cpu().numpy())
                    test_pe.append(grid.cpu().numpy())
    
    np.savez(os.path.join(model_dir, "train_latents.npz"), mu=np.array(train_mu), std=np.array(train_std))
    np.savez(os.path.join(model_dir, "val_latents.npz"), mu=np.array(val_mu), std=np.array(val_std))
    np.savez(os.path.join(model_dir, "test_latents.npz"), mu=np.array(test_mu), std=np.array(test_std))
    np.save(os.path.join(model_dir, "train_pe.npy"), np.array(train_pe))
    np.save(os.path.join(model_dir, "val_pe.npy"), np.array(val_pe))
    np.save(os.path.join(model_dir, "test_pe.npy"), np.array(test_pe))


def plot_flow_latents(epoch, flow, model_dir):
    train_latents = np.load(os.path.join(model_dir, "train_latents.npz"))
    mu, lv = train_latents['mu'], train_latents['std']
    train_samples = torch.distributions.normal.Normal(torch.Tensor(mu), torch.Tensor(lv).mul(0.5).exp()).sample()
    val_latents = np.load(os.path.join(model_dir, "val_latents.npz"))
    mu, lv = val_latents['mu'], val_latents['std']
    val_samples = torch.distributions.normal.Normal(torch.Tensor(mu), torch.Tensor(lv).mul(0.5).exp()).sample()
    test_latents = np.load(os.path.join(model_dir, "test_latents.npz"))
    mu, lv = test_latents['mu'], test_latents['std']
    test_samples = torch.distributions.normal.Normal(torch.Tensor(mu), torch.Tensor(lv).mul(0.5).exp()).sample()
    
    samples = []
    for i in range(5):
        with torch.no_grad():
            sample = flow.sample(1000).squeeze().cpu().numpy()
            if np.any(np.isnan(sample)):
                nan_location = np.unique(np.where(np.isnan(sample))[0])
                sample = np.delete(sample, nan_location, axis=0)
            samples.append(sample)
    sample = np.concatenate(samples, axis=0)

    pca = PCA(n_components=2)
    pca.fit(train_samples.squeeze().numpy())
    train_pca = pca.transform(train_samples.squeeze().numpy())
    sample_pca = pca.transform(sample)
    pca_dir = os.path.join(model_dir, "pca")
    if not os.path.exists(pca_dir):
        os.mkdir(pca_dir)

    #plot
    plt.figure(figsize=(10,10))
    plt.scatter(train_pca[:,0], train_pca[:,1], label="train")
    plt.scatter(sample_pca[:,0], sample_pca[:,1], label="sample")
    plt.legend()
    plt.savefig(os.path.join(pca_dir, f"train_pca_{epoch}.png"))
    plt.close()

    samples = []
    for i in range(3):
        with torch.no_grad():
            sample = flow.sample(500).squeeze().cpu().numpy()
            if np.any(np.isnan(sample)):
                nan_location = np.unique(np.where(np.isnan(sample))[0])
                sample = np.delete(sample, nan_location, axis=0)
            samples.append(sample)
    sample = np.concatenate(samples, axis=0)
        
    pca = PCA(n_components=2)
    pca.fit(val_samples.squeeze().numpy())
    val_pca = pca.transform(val_samples.squeeze().numpy())
    sample_pca = pca.transform(sample)

    #plot
    plt.figure(figsize=(10,10))
    plt.scatter(val_pca[:,0], val_pca[:,1], label="val")
    plt.scatter(sample_pca[:,0], sample_pca[:,1], label="sample")
    plt.legend()
    plt.savefig(os.path.join(pca_dir, f"val_pca_{epoch}.png"))
    plt.close()
        
    pca = PCA(n_components=2)
    pca.fit(test_samples.squeeze().numpy())
    test_pca = pca.transform(test_samples.squeeze().numpy())
    sample_pca = pca.transform(sample)

    #plot
    plt.figure(figsize=(10,10))
    plt.scatter(test_pca[:,0], test_pca[:,1], label="test")
    plt.scatter(sample_pca[:,0], sample_pca[:,1], label="sample")
    plt.legend()
    plt.savefig(os.path.join(pca_dir, f"test_pca_{epoch}.png"))
    plt.close()

    # distance to each point in val
    min_dist_idx = []
    for i in tqdm(sample):
        dist = i[None, :] - val_samples.squeeze().numpy()
        dist = np.linalg.norm(dist, axis=1)
        min_dist_idx.append(np.argmin(dist))
    print("val coverage:", np.unique(min_dist_idx).shape[0] / len(min_dist_idx))

    # distance to each point in test
    min_dist_idx = []
    for i in tqdm(sample):
        dist = i[None, :] - test_samples.squeeze().numpy()
        dist = np.linalg.norm(dist, axis=1)
        min_dist_idx.append(np.argmin(dist))
    print("test coverage:", np.unique(min_dist_idx).shape[0] / len(min_dist_idx))


def train_flow_prior(args, model_dir):
    # data
    unn_list = args.unn.split(",") if args.unn != "all" else "all"
    teeth_std = 9.8186 if unn_list == ['3'] else 11.75121  
    train_set = dataloader.Teeth_Dataset(unn=unn_list,
                                         folder_path=args.x_train,
                                         is_train=True,
                                         global_pc_std=teeth_std,
                                         # global std(norm(pc)) from fc_train_unn_3
                                         args=args)
    test_set = dataloader.Teeth_Dataset(unn=unn_list,
                                        folder_path=args.x_val,
                                        is_train=False,
                                        global_pc_std=teeth_std,  # global std(norm(pc)) from fc_train_unn_3
                                        args=args)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=args.num_workers,
                            pin_memory=False, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=args.num_workers,
                           pin_memory=False, collate_fn=collate_fn, drop_last=True)

    # model stuff
    vae = vfnet.Variational_autoencoder(args, num_points=2048).to(args.device)
    vae = utils.load_pretrained_model(vae, load_path=args.model_path)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    facets = pc_utils.facets_from_grid(75)
    edges = np.random.uniform(0.7,0.9, 4)
    meshgrid = [[-edges[0], edges[1], 75], [-edges[2], edges[3], 75]]
    x = np.linspace(*meshgrid[0])
    y = np.linspace(*meshgrid[1])
    grid = torch.from_numpy(np.array(list(itertools.product(x, y)))).unsqueeze(0)

    # Flow stuff
    num_layers, t = args.flow_n_layers , []
    for _ in range(num_layers):
        t.append(transforms.MaskedAffineAutoregressiveTransform(features=512, 
                                                                hidden_features=512,))
                                                                # num_blocks=3))
        t.append(transforms.ReversePermutation(features=512))

    transform = transforms.CompositeTransform(t)
    base_distribution = distributions.StandardNormal(shape=[512])
    flow = flows.Flow(transform=transform, distribution=base_distribution).to(args.device)

    # optimizer
    optimizer = torch.optim.Adam(flow.parameters(), lr=2e-4)

    for epoch in range(1, args.flow_num_epochs + 1):

        # train loop over dataloaders
        train_loss = 0
        recon_loss = 0

        for x in tqdm(train_loader):
            optimizer.zero_grad()
            pc_X, jaw_id = x["pc"], x["ids"]
            feature, feat1 = vae.encoder(pc_X.transpose(1,2).to(args.device))
            mu, lv = torch.chunk(feature, 2, dim=-1)
            z = vae.reparameterize(feature)
            batch_loss = -flow.log_prob(z.squeeze()).mean()
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
        
        train_loss /= len(train_loader)
        # print loss output to console in a nice manner
        print(f"Epoch {epoch} | Train loss {train_loss:.5f}")

        flow.eval()
        test_loss = 0
        recon_loss = 0
        for x in test_loader:
            pc_X, jaw_id = x["pc"], x["ids"]
            feature, feat1 = vae.encoder(pc_X.transpose(1,2).to(args.device))
            mu, lv = torch.chunk(feature, 2, dim=-1)
            z = vae.reparameterize(feature)
            with torch.no_grad():
                batch_loss = -flow.log_prob(z.squeeze()).mean()
            test_loss += batch_loss.item()
        
        if epoch % args.flow_viz_epochs == 0:
            plot_flow_latents(epoch, flow, model_dir)

            flow_sample_dir = os.path.join(model_dir, "samples")
            if not os.path.exists(flow_sample_dir):
                os.mkdir(flow_sample_dir)
            flow_sample_epoch_dir = os.path.join(flow_sample_dir, f"epoch_{epoch}")
            if not os.path.exists(flow_sample_epoch_dir):
                os.mkdir(flow_sample_epoch_dir)
            with torch.no_grad():
                sample = flow.sample(10)

            for i in range(10):
                with torch.no_grad():
                    latent = sample[i].unsqueeze(0).repeat(1,grid.shape[1],1).to(args.device)
                    output = vae.decoder.decode(latent, grid.float().to(args.device))
                    point_cloud_normalized = output['reconstruction'][:, :, :] * 9.8186
                    pc_utils.save_ply_manual(point_cloud_normalized.transpose(1,2).detach().cpu().numpy()[:, :, :].squeeze(),
                                            os.path.join(flow_sample_epoch_dir, 
                                            f"flow_sample_{i}.ply"),
                                            facets)
        
        test_loss /= len(test_loader)
        # print loss output to console in a nice manner
        print(f"Epoch {epoch} | Test loss {test_loss:.4f}")
    torch.save(flow.state_dict(), os.path.join(model_dir, "flow_prior.pth.tar"))


def train_point_encoding_predictor(args):
    # loggging dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.one_fold:
        model = OneFold().to(device)   
    else:
        model = TwoFold().to(device)
    chamf = pc_utils.ChamferLoss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    unn_list = args.unn.split(",") if args.unn != "all" else "all"
    teeth_std = 9.8186 if unn_list == ['3'] else 11.75121  
    train_set = dataloader.Teeth_Dataset(unn=unn_list,
                                         folder_path=args.x_train,
                                         is_train=True,
                                         global_pc_std=teeth_std,
                                         args=args)
    test_set = dataloader.Teeth_Dataset(unn=unn_list,
                                        folder_path=args.x_val,
                                        is_train=False,
                                        global_pc_std=teeth_std,  # global std(norm(pc)) from fc_train_unn_3
                                        args=args)

    data_loaders = {
        "Train": DataLoader(train_set, batch_size=32, shuffle=True, num_workers=args.num_workers,
                            pin_memory=False, collate_fn=collate_fn, drop_last=True),
        "Test": DataLoader(test_set, batch_size=32, shuffle=True, num_workers=args.num_workers,
                           pin_memory=False, collate_fn=collate_fn, drop_last=True)}

    model_dir = os.path.dirname(args.model_path)
    vae = vfnet.Variational_autoencoder(args, num_points=train_set.k).to(args.device)
    vae = utils.load_pretrained_model(vae, load_path=args.model_path)
    vae.eval()

    for epoch in range(1, args.pe_num_epochs + 1):
        train_loss, recon_loss = 0, 0

        for d in tqdm(data_loaders["Train"]):
            x= d['pc']
            optimizer.zero_grad()
            x = x.to(device) # weird technicality to get right shape
            feature, feat1 = vae.encoder(x.transpose(1,2))
            mu, lv = torch.chunk(feature, 2, dim=-1)
            z = vae.reparameterize(feature)
            z_repeated = z.repeat(1, feat1.shape[1], 1)
            cat = torch.cat((z_repeated, feat1), dim=-1)
            pe = vae.decoder.grid_map(cat) # (batch_size, num_points, 2)
            recon = model(z)
            loss = chamf(pe, recon)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        
        train_loss /= len(data_loaders['Train'])
        print(f"Epoch {epoch} |  Train loss {train_loss:.4f}")

        test_loss = 0
        recon_loss = 0
        for d in data_loaders["Test"]:
            with torch.no_grad():
                x= d['pc']
                optimizer.zero_grad()
                x = x.to(device)
                feature, feat1 = vae.encoder(x.transpose(1,2))
                mu, lv = torch.chunk(feature, 2, dim=-1)
                z = vae.reparameterize(feature)
                z_repeated = z.repeat(1, feat1.shape[1], 1)
                cat = torch.cat((z_repeated, feat1), dim=-1)
                pe = vae.decoder.grid_map(cat) # (batch_size, num_points, 2)
                recon = model(z)
                loss = chamf(pe.to(device), recon)
            test_loss += loss.item()
        
        test_loss /= len(data_loaders['Test'])
        # print loss output to console in a nice manner
        print(f"Epoch {epoch} | Test loss {test_loss:.4f}")

        pe_sample_dir = os.path.join(model_dir, "pe_samples")
        if not os.path.exists(pe_sample_dir):
            os.mkdir(pe_sample_dir)

        import matplotlib.pyplot as plt
        pe = pe.detach().cpu().numpy()
        recon = recon.detach().cpu().numpy()
        plt.close()
        fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
        ax[0].scatter(pe[0,:,0], pe[0,:,1])
        ax[0].set_title("gt")
        ax[1].scatter(recon[0,:,0], recon[0,:,1])
        ax[1].set_title("recon")
        plt.savefig(os.path.join(pe_sample_dir, f"pe_{epoch}_{args.pe_model_string}.png"))
    
    torch.save(model.state_dict(), os.path.join(model_dir, f"pe_{args.pe_model_string}.pth.tar"))


def train_point_encoding_flow(args):
    # loggging dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chamf = pc_utils.ChamferLoss()

    # Flow stuff
    num_layers, t = 5 , []
    for _ in range(num_layers):
        t.append(transforms.MaskedAffineAutoregressiveTransform(features=2, 
                                                                hidden_features=128,
                                                                context_features=512,
                                                                num_blocks=3))
        t.append(transforms.ReversePermutation(features=2))

    transform = transforms.CompositeTransform(t)
    base_distribution = distributions.StandardNormal(shape=[2])
    model = flows.Flow(transform=transform, distribution=base_distribution).to(args.device)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    unn_list = args.unn.split(",") if args.unn != "all" else "all"
    teeth_std = 9.8186 if unn_list == ['3'] else 11.75121  
    train_set = dataloader.Teeth_Dataset(unn=unn_list,
                                         folder_path=args.x_train,
                                         is_train=False,
                                         global_pc_std=teeth_std,
                                         args=args)
    test_set = dataloader.Teeth_Dataset(unn=unn_list,
                                        folder_path=args.x_val,
                                        is_train=False,
                                        global_pc_std=teeth_std,  # global std(norm(pc)) from fc_train_unn_3
                                        args=args)

    data_loaders = {
        "Train": DataLoader(train_set, batch_size=32, shuffle=True, num_workers=args.num_workers,
                            pin_memory=False, collate_fn=collate_fn, drop_last=True),
        "Test": DataLoader(test_set, batch_size=32, shuffle=True, num_workers=args.num_workers,
                           pin_memory=False, collate_fn=collate_fn, drop_last=True)}

    model_dir = os.path.dirname(args.model_path)
    vae = vfnet.Variational_autoencoder(args, num_points=train_set.k).to(args.device)
    vae = utils.load_pretrained_model(vae, load_path=args.model_path)
    vae.eval()

    for epoch in range(1, args.pe_num_epochs + 1):
        train_loss, recon_loss = 0, 0

        for d in tqdm(data_loaders["Train"]):
            x= d['pc']
            optimizer.zero_grad()
            x = x.to(device) # weird technicality to get right shape
            feature, feat1 = vae.encoder(x.transpose(1,2))
            mu, lv = torch.chunk(feature, 2, dim=-1)
            z = vae.reparameterize(feature)
            z_repeated = z.repeat(1, feat1.shape[1], 1)
            cat = torch.cat((z_repeated, feat1), dim=-1)
            pe = vae.decoder.grid_map(cat) # (batch_size, num_points, 2)
            # recon = model(z)
            # loss = -model.log_prob(pe.squeeze(), z_repeated).mean()
            loss = -model.log_prob(pe.reshape(-1, 2), z_repeated.reshape(-1, 512)).mean()
            # loss = chamf(pe, recon)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        
        train_loss /= len(data_loaders['Train'])
        print(f"Epoch {epoch} |  Train loss {train_loss:.4f}")

        test_loss = 0
        recon_loss = 0
        for d in data_loaders["Test"]:
            with torch.no_grad():
                x= d['pc']
                optimizer.zero_grad()
                x = x.to(device)
                feature, feat1 = vae.encoder(x.transpose(1,2))
                mu, lv = torch.chunk(feature, 2, dim=-1)
                z = vae.reparameterize(feature)
                z_repeated = z.repeat(1, feat1.shape[1], 1)
                cat = torch.cat((z_repeated, feat1), dim=-1)
                pe = vae.decoder.grid_map(cat) # (batch_size, num_points, 2)
                # recon = model(z)
                # loss = chamf(pe.to(device), recon)
                loss = -model.log_prob(pe.reshape(-1, 2), z_repeated.reshape(-1, 512)).mean()

            test_loss += loss.item()
        
        test_loss /= len(data_loaders['Test'])
        # print loss output to console in a nice manner
        print(f"Epoch {epoch} | Test loss {test_loss:.4f}")

        import matplotlib.pyplot as plt
        pe = pe.detach().cpu().numpy()
        with torch.no_grad():
            recon = model.sample(1, context=z_repeated.reshape(-1,512)).squeeze().detach().cpu().numpy()
        recon = recon.reshape(32, 2048, 2)
        print("batch chamf:", chamf(torch.Tensor(pe), torch.Tensor(recon)))

        plt.close()
        fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
        ax[0].scatter(pe[0,:,0], pe[0,:,1])
        ax[0].set_title("gt")
        ax[1].scatter(recon[0,:,0], recon[0,:,1])
        ax[1].set_title("recon")
        plt.savefig(os.path.join(model_dir, f"pe_{epoch}_flow.png"))
    
    torch.save(model.state_dict(), os.path.join(model_dir, "pe_flow.pth.tar"))


def load_flow_prior(args):
    model_dir = os.path.dirname(args.model_path)

    num_layers, t = args.flow_n_layers, []
    for _ in range(num_layers):
        t.append(transforms.MaskedAffineAutoregressiveTransform(features=512, 
                                                            hidden_features=512))
                                                            # num_blocks=3))
        t.append(transforms.ReversePermutation(features=512))
    transform = transforms.CompositeTransform(t)
    base_distribution = distributions.StandardNormal(shape=[512])
    flow = flows.Flow(transform=transform, distribution=base_distribution).to(args.device)

    flow_state_dict = torch.load(os.path.join(model_dir, "flow_prior.pth.tar"))
    flow.load_state_dict(flow_state_dict)
    flow.eval()
    
    plot_flow_latents("final", flow, model_dir)

    return flow


def get_ref(path, args):
    unn_list = args.unn.split(",") if args.unn != "all" else "all"
    teeth_std = 9.8186 if unn_list == ['3'] else 11.75121  
    test_set = dataloader.Teeth_Dataset(unn=unn_list,
                                        folder_path=path,
                                        is_train=False,
                                        global_pc_std=teeth_std,  # global std(norm(pc)) from fc_train_unn_3
                                        args=args)

    data_loaders = {
        "Test": DataLoader(test_set, batch_size=1, shuffle=True, num_workers=args.num_workers,
                           pin_memory=False, collate_fn=collate_fn, drop_last=True)}

    all_ref = []
    for phase in ["Test"]:
        for idx, d in enumerate(tqdm(data_loaders[phase])):    
            pc_X = d['pc']
            all_ref.append(pc_X)

    return torch.cat(all_ref,dim=0)*teeth_std


def get_samples(args, num_samples, flow, pe):
    print("Generating samples..."   )
    # model stuff
    vae = vfnet.Variational_autoencoder(args, num_points=2048).to(args.device)
    vae = utils.load_pretrained_model(vae, load_path=args.model_path)
    vae.eval()
    unn_list = args.unn.split(",") if args.unn != "all" else "all"
    teeth_std = 9.8186 if unn_list == ['3'] else 11.75121  
    

    all_samples = []
    for _ in tqdm(range(num_samples)):
        with torch.no_grad():
            z = flow.sample(1)
            point_encoding = pe(z, add_n_points=23)
            z = z.repeat(1, point_encoding.shape[1], 1)
            sample = vae.decoder.decode(z, point_encoding)
            # pc_utils.save_pointcloud(sample['reconstruction'].cpu().numpy()*9,
            #                          os.path.join("./results", f"sample_{_}.ply"))
            all_samples.append(sample['reconstruction'])
    return torch.cat(all_samples, dim=0)*teeth_std

if __name__ == "__main__":
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.dirname(args.model_path)
    assert args.test_name is not None, "Test name must be specified"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    f = open(os.path.join(model_dir, "commandline_input.json"), "r")
    commandline_input = json.load(f)
    for k, v in commandline_input.items():
        if k not in args:
            vars(args)[k] = v

    # save copy of this code into test folder:
    test_dir = "./test"
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)
    
    # copy code
    shutil.copy("./train_sampling.py", os.path.join(test_dir, args.test_name+ ".py"))

    latent_files_not_generated = False
    for model_file in ["train_latents.npz", "val_latents.npz", "test_latents.npz", "train_pe.npy", "val_pe.npy", "test_pe.npy"]:
        if not os.path.exists(os.path.join(model_dir, model_file)):
            latent_files_not_generated = True
            break
    if latent_files_not_generated:
        print("Latent files not found... Generating new latent files")
        get_latents(args, model_dir)
        print("Latent files generation done")
    else:
        print("Latent files found")
    
    if not os.path.exists(os.path.join(model_dir, "flow_prior.pth.tar")):
        print("Flow prior not found... Training new flow prior")
        train_flow_prior(args, model_dir)
        flow_prior = load_flow_prior(args)
    else:
        flow_prior = load_flow_prior(args)
        print("Existing flow prior loaded")

    args.pe_model_string = "oneFold" if args.one_fold else "twoFold"
    if not os.path.exists(os.path.join(model_dir, f"pe_{args.pe_model_string}.pth.tar")):
        train_point_encoding_predictor(args)
        # train_point_encoding_flow(args)
        if args.one_fold:
            pe = OneFold().to(args.device)
        else:
            pe = TwoFold().to(args.device)
        pe.load_state_dict(torch.load(os.path.join(model_dir, f"pe_{args.pe_model_string}.pth.tar")))
    else:
        if args.one_fold:
            pe = OneFold().to(args.device)
        else:
            pe = TwoFold().to(args.device)
        pe.load_state_dict(torch.load(os.path.join(model_dir, f"pe_{args.pe_model_string}.pth.tar")))
        print("Existing point encoding predictor loaded")
    
    all_ref = get_ref(args.x_test, args).transpose(1,2).to(args.device)
    all_samples = get_samples(args, all_ref.shape[0], flow_prior, pe).transpose(1,2).to(args.device)

    # for size sanity
    # pc_utils.save_pointcloud(all_ref[:1].cpu().numpy(), "./test_ref.ply")
    # pc_utils.save_pointcloud(all_samples[:1].cpu().numpy(), "./test.ply")

    print(args.test_name)
    results = metrics.compute_all_metrics(all_samples, all_ref, batch_size=256, accelerated_cd=True)
    print(args.test_name)

    print(results)
    for k, v in results.items():
        print(k, v*100)
    print("lol")
    
    results['args'] = vars(args)
    # # save results
    torch.save(results, os.path.join(model_dir, "results.pt"))

    # # go through dict and change tensors to list
    # for k, v in results.items():
    #     if isinstance(v, torch.Tensor):
    #         results[k] = v.tolist()
    # results['args']['device'] = "cuda"
    # # save dict as json
    # with open(os.path.join(model_dir, "test_results.json"), "w") as f:
    #     json.dump(results, f, indent=4)
