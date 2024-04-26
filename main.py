import argparse
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import dataloader
import utils
from model import vfnet
from train import train

parser = argparse.ArgumentParser(description='Training loop boiler plate')
parser.add_argument('--dataset', required=False, default="teeth", help="Data set, pick from ['teeth', 'modelnet40', 'shapenetcore']")
parser.add_argument('--x_train', required=False, help='Path to X_train folder', dest="x_train")
parser.add_argument('--x_val', required=False, help='Path to X_est folder', dest="x_test")
parser.add_argument("--point_normals", action="store_true", default=False, help="If point normals should be included in data")
parser.add_argument("--std_training", action="store_true", default=False, help="trainig std only")
parser.add_argument("--unn", required=False, default="3", help="unns to include, separate with comma and no space")
parser.add_argument("--shapenet_single_class", action="store_true", default=False, help="If point normals should be included in data")
parser.add_argument('--num_workers', required=False, type=int, default=16)
parser.add_argument("--sn_anneal", action="store_true", default=False, help="If surface normal loss should be annealed")
parser.add_argument("--cpu_only", action="store_true", default=False, help="force into cpu mode, used for debugging")
parser.add_argument("--model", required=False, type=str, default="vae", help="model choice")
parser.add_argument('--resume_from', required=False, default="", help='Path to state dict to continue training from')
parser.add_argument("--encoder", required=False, type=str, default="foldnet", help="Foldingnet encoder")
parser.add_argument("--decoder", required=False, type=str, default="stochman", help="stochman decoder")
parser.add_argument("--point_encoding", action="store_true", default=False, help="If model should map to grid or use entire grid")
parser.add_argument('--iwae_k', required=False, type=int, default=1)
parser.add_argument('--static_kl', required=False, type=float, default=0)
parser.add_argument('--scaling_std', action="store_true", required=False, default=False)
parser.add_argument("--exp_name", required=False, type=str, default=None,
                    help="Name for experiment, default being datetime")
parser.add_argument("--commit_name", required=False, type=str, default=None)
parser.add_argument("--commit_text", required=False, type=str, default=None)
parser.add_argument("--rundatetime", required=False, type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))


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
    return dict(pc=pc, ids=ids)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    # settings:
    utils.add_standard_args_to_parser(parser)
    args = parser.parse_args()
    device = "cpu" if args.cpu_only else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loaders = None

    # data stuff
    if args.dataset.lower() in ['modelnet40', 'shapenetcorev2']:
        train_set, val_set, data_loaders = dataloader.build(args)
        collate_fn = None
    elif args.dataset.lower() == "teeth":
        unn_list = args.unn.split(",") if args.unn != "all" else "all"
        teeth_std = 9.8186 if unn_list == ['3'] else 11.75121  # 12.425647 is for all unn 3
        args.teeth_std = teeth_std
        train_set = dataloader.Teeth_Dataset(unn=unn_list,
                                             folder_path=args.x_train,
                                             is_train=True,
                                             global_pc_std=teeth_std, #2*2.185 # global std(norm(pc)) from fc_train_unn_3
                                             args=args)
        test_set = dataloader.Teeth_Dataset(unn=unn_list,
                                            folder_path=args.x_test,
                                            is_train=False,
                                            global_pc_std=teeth_std, # 2.185, # global std(norm(pc)) from fc_train_unn_3
                                            args=args)
    elif args.dataset.lower() == "shapenet15":
        train_set, test_set, data_loaders = dataloader.build(args)
    else:
        raise ValueError(f"Dataset not recognized: {args.dataset}. Available options: ['teeth', 'modelnet40', 'shapenetcorev2']")

    # model stuff
    model = vfnet.Variational_autoencoder(args, num_points=train_set.k, global_std=train_set.global_pc_std).to(device)

    # optimizer and scheduler
    optimizer = optim.Adamax(model.parameters(), args.lr, [args.beta1, args.beta2])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, verbose=True, factor=0.85)

    #  model loading
    epoch = None
    if len(args.resume_from) > 0:  # to lazy to make model and optimizer load together
        model = utils.load_pretrained_model(model, load_path=args.resume_from)
        epoch, optimizer_state_dict, best_loss = utils.load_previous_training_params(args.resume_from, optimizer)
        optimizer = optim.Adamax(model.parameters(), args.lr, [args.beta1, args.beta2]) # tmp to be deleted

        if args.std_training:
            model.decoder.init_std(device)
            # model = utils.load_pretrained_model(model, load_path=args.resume_from)
            optimizer = optim.Adam(model.decoder.std.parameters(), args.lr, [args.beta1, args.beta2])
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, verbose=True,
                                                                      factor=0.85)
            epoch = 0
        else:
            optimizer.load_state_dict(optimizer_state_dict)  # stupid in place operation

    # test_batch_size = 2 if args.dataset.lower() == "teeth" else args.batch_size
    if data_loaders is None:
        data_loaders = {
            "Train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                pin_memory=True, collate_fn=collate_fn, drop_last=True),
            "Test": DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               pin_memory=True, collate_fn=collate_fn, drop_last=True)}

    train(data_loaders=data_loaders,
          model=model,
          optimizer=optimizer,
          scheduler=lr_scheduler,
          device=device,
          args=args,
          preloaded_epoch=epoch)



