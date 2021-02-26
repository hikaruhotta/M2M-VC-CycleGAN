"""
Utils of the asr pipeline.
Adapted from https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO
"""

import torch
import os

class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def GreedyDecoder(output, text_transform, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(
            labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets


def save_ckpt(epoch, model, model_name, optimizer, lr_scheduler, ckpt_dir, device):
    """
    Save model checkpoint to disk.

    Args:
        epoch (int): Current epoch
        model : Model to save
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler for optimizer
        optimizer (torch.optim.Optimizer): Optimizer for model parameters
        device (str): Device where the model/optimizer parameters belong
        model_name (str): Name of model to save
        ckpt_dir (str): Directory to save the checkpoint
    """
    # Unwrap nn.DataParallel module if needed
    try:
        model_class = model.module.__class__.__name__
        model_state = model.to('cpu').module.state_dict()
        print("Unwrapped DataParallel module.")
    except AttributeError:
        model_class = model.__class__.__name__
        model_state = model.to('cpu').state_dict()

    ckpt_dict = {
        'ckpt_info': {'epoch': epoch},
        'model_class': model_class,
        'model_state': model_state,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }

    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_epoch_{epoch}.pth.tar")
    torch.save(ckpt_dict, ckpt_path)
    model.to(device)
    print(f"Saved {model_name} at epoch {epoch} to {ckpt_path}.")
