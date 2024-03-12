import os
import shutil

import torch
import torch.nn as nn


def add_standard_args_to_parser(parser):
    """
    Adding rarely changes options to parser.
    They remain as arguments to be easily changed if necessary / saved each run
    """
    parser.add_argument('--k', type=int, default=None, metavar='N', help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--feat_dims', type=int, default=512, metavar='N', help='Number of dims for feature ')
    parser.add_argument("--fold_orig_shape", required=False, type=str, default="plane")
    parser.add_argument("--num_epochs", required=False, type=int, default=15000)
    parser.add_argument('--beta1', required=False, type=float, default=0.9)
    parser.add_argument('--beta2', required=False, type=float, default=0.999)
    parser.add_argument("--patience", required=False, type=int, default=50)
    parser.add_argument('--lr', required=False, type=float, default=0.001)
    parser.add_argument('--batch_size', required=False, type=int, default=64)
    return parser


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def save_checkpoint(file_path, epochs, model, optimizer, scheduler=None, description=None):
    """
    Saves epoch, model, optimizer for further training
    :param file_path: Path and filename to save file
    :param epochs: current epoch
    :param model: model
    :param optimizer:
    :param scheduler:
    :return:
    """
    state = {'epoch': epochs, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    if scheduler is not None:
        state["scheduler"] = scheduler
    if description is not None:
        state["description"] = description
    torch.save(state, file_path)


def load_checkpoint(ckpt_path, model, optimizer, scheduler=None):
    """
    Loads checkpoint with optimizer and scheduler to resume training
    :param optimizer:
    :param ckpt_path:
    :return:
    """
    checkpoint = torch.load(ckpt_path)
    print("model description:", checkpoint.get("description"))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["epoch"], model, optimizer, scheduler


def save_pretraining(file_path, epoch, model, optimizer, best_loss):
    """ save checkpoint in case of crash """
    state = {'epoch': epoch, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'best_loss': best_loss}
    torch.save(state, file_path)


def load_previous_training_params(file_path, optimizer):
    """ loads checkpoint in case of crash ret"""
    state = torch.load(file_path)

    return (state['epoch'],
            state['optimizer'],
            state['best_loss'])


def load_pretrained_model(model, load_path=None):
    model_dict = model.state_dict()
    loaded_state_dict = torch.load(load_path)["state_dict"]
    for k in loaded_state_dict:
        if k in model_dict:
            model_dict[k] = loaded_state_dict[k]
            print("    Found weight: " + k)
        else:
            print(f"WARNING: Expected weights {k} not found")
        model.load_state_dict(model_dict)
    return model


def dir_cleaner(folder, file_format_to_delete):
    """
    If directory exists, delete all files with input format.
    else create dir
    :return:
    """
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if file_path[-len(file_format_to_delete):] == file_format_to_delete:
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.mkdir(os.path.join(folder))