import torch
import torch.distributions as td
import torch.nn as nn
import numpy as np

from model.decoders import FoldNet_Decoder_stochman, Decoder_Linear
from model.encoders import FoldNet_Encoder_Linear
from pc_utils import ChamferLoss


class ReconstructionNet(nn.Module):
    def __init__(self, args, num_points):
        super(ReconstructionNet, self).__init__()
        self.input_dim = 6 if args.point_normals else 3

        self.encoder = FoldNet_Encoder_Linear(args)
        self.decoder = Decoder_Linear(args, num_points)

        self.loss = ChamferLoss(args)

    def forward(self, input, sample_grid=False, edge_only=False, jacobian=False):
        if input.shape[2] != self.input_dim and input.shape[1] == self.input_dim:
            input = input.transpose(1, 2)
        feature, feat1 = self.encoder(input)
        if jacobian:
            return self.decoder(feature, feat1, sample_grid, edge_only, jacobian)
        else:
            return self.decoder(feature, feat1, sample_grid, edge_only)

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, epoch, ground_truth, model_output, iwae, std=1):
        """ iwae because the vae needs it """
        assert isinstance(model_output, dict)
        reconstruction = model_output["reconstruction"]
        self.loss.current_epoch = epoch

        # dim checks
        if ground_truth.shape[-1] > 6:
            ground_truth = ground_truth.transpose(1, 2)
        if reconstruction.shape[-1] != 3:
            reconstruction = reconstruction.transpose(1, 2)
        ground_truth_vertex = ground_truth[:, :, :3]

        #  dirty surface normal calculations
        if "jacobian" in model_output.keys():
            bs, n_dim, n_points = ground_truth.shape
            reconstruction = reconstruction.view(bs, -1, 3)

            ground_truth_normals = torch.nn.functional.normalize(ground_truth[:, :, 3:], dim=-1)
            jac = model_output['jacobian']
            jac = jac.squeeze()  # (bs, n_point, 3, 2)
            surface_normals = torch.cross(jac[:, :, 1], jac[:, :, 0], dim=-1)
            surface_normals = torch.nn.functional.normalize(surface_normals, dim=-1)
            reconstructed_surface_normals = surface_normals.view(bs, -1, 3)

            # tmp
            # ground_truth_vertex = ground_truth
            return self.loss(ground_truth_vertex*std, reconstruction*std, (ground_truth_normals, reconstructed_surface_normals))
        else:
            d = self.loss(ground_truth_vertex*std, reconstruction*std)
            # euclidean distance
            d['total_loss'] = (ground_truth_vertex*std - reconstruction*std).pow(2).sum(-1).mean()
            return d



class Variational_autoencoder(nn.Module):
    def __init__(self, args, num_points, global_std=8.18936):
        super(Variational_autoencoder, self).__init__()
        self.input_dim = 6 if args.point_normals else 3
        self.encoder = FoldNet_Encoder_Linear(args)
        self.decoder = Decoder_Linear(args, num_points)
        self.loss = ChamferLoss(args)
        self.max_epochs = args.num_epochs
        self.feat_dims = args.feat_dims
        self.static_kl = args.static_kl
        self.scaling_std = args.scaling_std
        self.prior = None
        self.backprop_step = 0
        self.iwae_k = args.iwae_k
        self.warm_up_epochs = int(args.num_epochs / 4)
        self.surface_normal_loss_direction = None
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

    def get_loss(self, epoch, ground_truth, model_output, iwae=False, std=1):
        reconstruction = model_output["reconstruction"]
        bs, n_dim, n_points = ground_truth.shape
        if reconstruction.shape[-1] == 3:  # OBS: THIS MAY FUCK UP CONVERGENCE IF DONE WRONG
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
            # scaling = 0.000005
            # scaling = 0.0001 - 0.00005 * min(1., epoch / self.warm_up_epochs)
            self.std = torch.ones_like(reconstruction) * np.sqrt(scaling)
        else:
            self.std = (model_output["std"].repeat(1,1,3).mul(0.5).exp() * 0.0005)  + 1e-10

        if self.iwae_k != 1 and iwae:
            self.iwae_loss(ground_truth)

        if "std" not in model_output.keys():
            p_xGz = td.studentT.StudentT(df=3, loc=reconstruction, scale=self.std)
        else:
            p_xGz = td.normal.Normal(loc=reconstruction, scale=self.std)

        kl = td.kl_divergence(self.q_zGx, self.prior).sum(-1).sum(-1)
        recon_error = p_xGz.log_prob(ground_truth_vertex).sum(-1).sum(-1)
        kl_coeff = min(1., epoch / self.warm_up_epochs) if self.static_kl == 0 else self.static_kl
        ELBO = recon_error - kl * kl_coeff

        #  dirty surface normal calculations
        if False:
            bs, n_dim, n_points = ground_truth.shape
            reconstruction = reconstruction.view(bs, -1, 3)

            ground_truth_normals = torch.nn.functional.normalize(ground_truth[:, :, 3:], dim=-1)
            jac = model_output['jacobian']
            jac = jac.squeeze()  # (bs, n_point, 3, 2)

            if self.surface_normal_loss_direction in [None,1]:
                surface_normals = torch.cross(jac[:, :, 0], jac[:, :, 1], dim=-1)
                surface_normals = torch.nn.functional.normalize(surface_normals, dim=-1)
                reconstructed_surface_normals = surface_normals.view(bs, -1, 3)

                loss_1 = self.loss(ground_truth_vertex, reconstruction,
                                   (ground_truth_normals, reconstructed_surface_normals))

            if self.surface_normal_loss_direction in [None,2]:
                surface_normals = torch.cross(jac[:, :, 1], jac[:, :, 0], dim=-1)
                surface_normals = torch.nn.functional.normalize(surface_normals, dim=-1)
                reconstructed_surface_normals = surface_normals.view(bs, -1, 3)
                loss_2 = self.loss(ground_truth_vertex, reconstruction,
                                   (ground_truth_normals, reconstructed_surface_normals))

            if self.surface_normal_loss_direction is None:
                surface_normal_loss = min(loss_1['surface_normal_angle'].mean(), loss_2['surface_normal_angle'].mean())
                if epoch > int(self.max_epochs / 4):
                    if loss_1['surface_normal_angle'].mean() < loss_2['surface_normal_angle'].mean():
                        self.surface_normal_loss_direction = 2
                    else:
                        self.surface_normal_loss_direction = 1

            sn_loss_coeff = max(1, (100 - min(1, epoch / self.warm_up_epochs) * 100))  # scaling down towards 0.01

            return {"total_loss": -ELBO.mean() + surface_normal_loss*sn_loss_coeff,
                    "chamfer": loss_1['chamfer'],
                    "surface_normal_angle": surface_normal_loss,
                    "elbo": -ELBO.mean(),
                    "kl": kl.mean(),
                    "kl_coeff": kl_coeff,
                    "sn_loss_coeff": sn_loss_coeff}
        else:
            loss_1 = self.loss(ground_truth_vertex*std, reconstruction*std)
            # punish weird grids
            # n_points = self.decoder.grid.shape[1]
            # sorted = self.decoder.grid.sort(dim=1)[0]
            # C = torch.linspace(0, 1, n_points, requires_grad=False).unsqueeze(0).unsqueeze(-1).repeat(self.decoder.grid.shape[0], 1, 2).to(
            #     self.decoder.grid.device)
            # a = self.decoder.grid.min(dim=1)[0].unsqueeze(1).repeat(1, n_points, 1).to(self.decoder.grid.device)
            # b = self.decoder.grid.max(dim=1)[0].unsqueeze(1).repeat(1, n_points, 1).to(self.decoder.grid.device)
            # D = (((sorted - a) / (b - a)) - C).pow(2).sum(dim=[0, 1])

            return {"total_loss": -(ELBO).mean(),  # + (D.mean() * 1e-4),
                    "chamfer": loss_1['chamfer'],
                    "elbo": -ELBO.mean(),
                    # "uniform_loss": D.mean(),
                    "kl": kl.mean(),
                    "kl_coeff": kl_coeff}

    def linear_decrease(self):
        return 1 - 0.99 * (self.current_epoch / self.max_epochs)

    def iwae_loss(self, ground_truth):

        log_px_z = None
        ground_truth_vertex = ground_truth[:, :, :3]
        k = torch.Tensor([self.iwae_k]).to(ground_truth.device)
        kld = td.kl_divergence(self.q_zGx, self.prior).mean(-1).repeat(1, self.iwae_k)
        for i in range(self.iwae_k):
            z = self.q_zGx.rsample()
            model_output_k = self.decoder(z, ground_truth, sample_grid=False, edge_only=False)
            p_xGz = td.multivariate_normal.MultivariateNormal(loc=model_output_k['reconstruction'].transpose(1, 2),
                                                              covariance_matrix=self.covariance)
            if log_px_z is None:
                log_px_z = p_xGz.log_prob(ground_truth_vertex).mean(-1).unsqueeze(1)
            else:
                log_px_z = torch.cat([log_px_z, p_xGz.log_prob(ground_truth_vertex).mean(-1).unsqueeze(1)], 1)

        log_wk = log_px_z - kld
        if log_wk.sum() > 0:
            log_wk *= -1
        ELBO = log_wk.logsumexp(dim=-1) - k.log()
        loss_1 = self.loss(ground_truth_vertex, model_output_k['reconstruction'].transpose(1, 2))
        return {"total_loss": -ELBO.mean(),
                "chamfer": loss_1['chamfer'],
                "elbo": -ELBO.mean(),
                "kl": kld.mean(),
                "iwae_k": self.iwae_k}
