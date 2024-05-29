import torch
import torch.distributions as td
import torch.nn as nn
import numpy as np

from model.decoders import Decoder_Linear
from model.encoders import FoldNet_Encoder_Linear
from pc_utils import ChamferLoss


class Variational_autoencoder(nn.Module):
    def __init__(self, args, num_points, global_std=8.18936):
        super(Variational_autoencoder, self).__init__()
        self.input_dim = 6 if args.point_normals else 3
        self.encoder = FoldNet_Encoder_Linear(args)
        self.decoder = Decoder_Linear(args, num_points)
        self.loss = ChamferLoss()
        self.max_epochs = args.num_epochs
        self.feat_dims = args.feat_dims
        self.prior = None
        self.warm_up_epochs = int(args.num_epochs / 4)
        self.global_std = global_std

    def reparameterize(self, feature):
        mu, lv = torch.chunk(feature, 2, dim=-1)
        self.q_zGx = td.normal.Normal(loc=mu, scale=lv.mul(0.5).exp() + 1e-10)
        return self.q_zGx.rsample()

    def forward(self, input_pc, sample_grid=False, edge_only=False, jacobian=False):
        if input_pc.shape[2] != self.input_dim and input_pc.shape[1] == self.input_dim:
            input_pc = input_pc.transpose(1, 2)
        feature, feat1 = self.encoder(input_pc)
        latent_codes = self.reparameterize(feature)
        if jacobian:
            return self.decoder(latent_codes, feat1, eval, edge_only, jacobian)
        else:
            return self.decoder(latent_codes, feat1, eval, edge_only)
        
    def get_latent(self, pc):
        if pc.shape[2] != self.input_dim and pc.shape[1] == self.input_dim:
            pc = pc.transpose(1, 2)
        feature, feat1 = self.encoder(pc)
        return self.reparameterize(feature)
    
    def get_grid(self, pc):
        if pc.shape[2] != self.input_dim and pc.shape[1] == self.input_dim:
            pc = pc.transpose(1, 2)
        feature, feat1 = self.encoder(pc)
        latent_codes = self.reparameterize(feature)
        latent_codes_repeated = latent_codes.repeat(1, feat1.shape[1], 1)
        cat = torch.cat((latent_codes_repeated, feat1), dim=-1)
        return self.decoder.grid_map(cat)

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, epoch, ground_truth, model_output, std=1):
        reconstruction = model_output["reconstruction"]
        bs, n_dim, n_points = ground_truth.shape
        if reconstruction.shape[-1] == 3:  # OBS: THIS MAY GO WRONG CONVERGENCE IF DONE WRONG
            reconstruction = reconstruction.reshape(bs, -1, 3)
        self.loss.current_epoch, self.current_epoch = epoch, epoch

        # dim checks
        if ground_truth.shape[-1] > 6:
            ground_truth = ground_truth.transpose(1, 2)
        if reconstruction.shape[-1] != 3:
            reconstruction = reconstruction.transpose(1, 2)
        ground_truth_vertex = ground_truth[:, :, :3]

        if self.prior is None or self.prior_bs != ground_truth.shape[0]:
            self.prior_bs = ground_truth.shape[0]
            self.prior = td.normal.Normal(
                loc=torch.zeros_like(self.q_zGx.loc, requires_grad=False),
                scale=torch.ones_like(self.q_zGx.scale, requires_grad=False))
        if "std" not in model_output.keys():
            scaling = 0.0005
            self.std = torch.ones_like(reconstruction) * np.sqrt(scaling)
        else:
            self.std = (model_output["std"].repeat(1,1,3).mul(0.5).exp() * 0.0005)  + 1e-10

        if "std" not in model_output.keys():
            p_xGz = td.studentT.StudentT(df=3, loc=reconstruction, scale=self.std)
        else:
            p_xGz = td.normal.Normal(loc=reconstruction, scale=self.std)

        kl = td.kl_divergence(self.q_zGx, self.prior).sum(-1).sum(-1)
        recon_error = p_xGz.log_prob(ground_truth_vertex).sum(-1).sum(-1)
        kl_coeff = min(1., epoch / self.warm_up_epochs)
        ELBO = recon_error - kl * kl_coeff

        loss_1 = self.loss(ground_truth_vertex*std, reconstruction*std)

        return {"total_loss": -(ELBO).mean(),  # + (D.mean() * 1e-4),
                "chamfer": loss_1['chamfer'],
                "elbo": -ELBO.mean(),
                # "uniform_loss": D.mean(),
                "kl": kl.mean(),
                "kl_coeff": kl_coeff}

    def linear_decrease(self):
        return 1 - 0.99 * (self.current_epoch / self.max_epochs)