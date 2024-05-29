import json
from datetime import datetime
from pathlib import Path

import git
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pc_utils
import utils


def loss_print(epoch, phase, epoch_loss_dict, std):
    """ printing epoch loss from epoch_loss_dict """
    print(f"epoch: {epoch}, phase {phase}", end="")
    for k, v in epoch_loss_dict.items():
        if k == "num_samples":
            continue
        elif "coeff" in k:
            print(f", {k.capitalize()}: {v[0]:.2f}", end="")
            continue
        elif "chamfer" in k:
            v = torch.Tensor(v).mean()
            print(f", {k.capitalize()}: {v:.2f} ({float(v) * float(std):.2f})", end="")
            continue
        elif isinstance(v, list):
            v = torch.Tensor(v).mean()
            print(f", {k.capitalize()}: {v:.2f}", end="")
    print("")  # for newline


def save_epoch_loss(epoch_loss_dict, backprop_loss_dict):
    """ appending loss to dict key """
    if len(epoch_loss_dict) == 1:
        for k, v in backprop_loss_dict.items():
            epoch_loss_dict[k] = [float(v)]
    else:
        for k, v in backprop_loss_dict.items():
            if "coeff" in k:  # no need to append coeffs
                continue
            epoch_loss_dict[k].append(float(v))
    return epoch_loss_dict


def tensorboard_save(writer, phase, epoch, epoch_loss_dict, std):
    """ saving to tensorboard """
    for k, v in epoch_loss_dict.items():
        if k == "total_loss":
            loss_stack = torch.Tensor(v)
            writer.add_scalar(f"foldingnet/{phase}_epoch", loss_stack.sum() / epoch_loss_dict['num_samples'], epoch)
            # writer.add_histogram(f"foldingnet/{phase}_hist_epoch", loss_stack.detach().cpu().numpy(), epoch)  # dead dunno why
        elif k in ["num_samples"]:
            continue
        elif "chamfer" in k:
            v = torch.Tensor(v).mean()
            writer.add_scalar(f"{k.capitalize()}/{phase}", v*std, epoch)
            continue
        else:
            if isinstance(v, list):
                v = torch.Tensor(v).mean()
            writer.add_scalar(f"{k.capitalize()}/{phase}", v, epoch)


def train(data_loaders, model, optimizer, scheduler, device, args, preloaded_epoch=None):
    # loggging dir
    best_loss = np.inf
    runs = "runs" # if save loc modification needed:
    args.current_commit = git.Repo(search_parent_directories=True).head.object.hexsha[:7]
    args.commit_text = git.Repo(search_parent_directories=True).head.reference.commit.message
    log_dir = Path(runs, args.current_commit,
                   datetime.now().strftime("%Y%m%d_%H%M%S")) if args.exp_name is None else Path(
        runs, args.current_commit, args.exp_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    with open(f"{log_dir}/commandline_input.json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    adjust_epoch = 0 if preloaded_epoch is None else preloaded_epoch
    for epoch in range(adjust_epoch + 1, args.num_epochs + 1):

        for phase in ['Train', 'Test']:
            if epoch < int(args.num_epochs / 4):
                if epoch % 10 != 0 and phase == 'Test':
                    continue

            epoch_loss_dict = {"num_samples": 0}

            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for idx, d in enumerate(tqdm(data_loaders[phase])):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'Train'):
                    pc_X, _ = d["pc"], d["ids"]
                    pc_X = pc_X.to(device)

                    # TODO: Try gradient clipping
                    output_dict = model(pc_X, jacobian=(args.decoder.lower() == "stochman" and args.point_normals))
                    loss_dict = model.get_loss(epoch, pc_X, output_dict)
                    if phase == "Train":
                        loss_dict["total_loss"].backward()
                        model.loss.backprop_num += 1
                        optimizer.step()

                    if "std" in output_dict:
                        loss_dict['std_corr'] = pc_utils.loss_std_correlation(pc_X, output_dict).mean()

                    # logging
                    epoch_loss_dict = save_epoch_loss(epoch_loss_dict, loss_dict)
                    epoch_loss_dict["num_samples"] += pc_X.shape[0]
                    del output_dict, loss_dict

            epoch_loss = torch.Tensor(epoch_loss_dict['total_loss']).mean()
            loss_print(epoch, phase, epoch_loss_dict, data_loaders[phase].dataset.global_pc_std)
            tensorboard_save(writer, phase, epoch, epoch_loss_dict, data_loaders[phase].dataset.global_pc_std)

            if phase == "Test":
                save_iter = 1000 if args.num_epochs > 10000 else 100
                if epoch % save_iter == 0:
                    utils.save_pretraining(f"{log_dir}/epoch_{epoch}.pth.tar", epoch, model, optimizer, best_loss)

                scheduler.step(epoch_loss)
