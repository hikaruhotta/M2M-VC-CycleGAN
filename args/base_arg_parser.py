"""
Base arguments for all scripts
"""

import argparse
import json
import os
from datetime import datetime
import torch
import numpy as np
import random


class BaseArgParser(object):
    """
    Base argument parser for args shared between test and train modes.

    ...
    Attributes
    ----------
    parser : argparse.ArgumentParser
        ArgumentParser object used to parse command line args

    Methods
    -------
    parse_args():
        Parse arguments, create checkpoints and vizualization directory, and sets up gpu device

    print_options(args):
        Prints and save options
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description=' ')
        self.isTrain = True

        self.parser.add_argument('--debug', default=False, action='store_true',
                                 help=('Whether to run code deterministically.'))
        self.parser.add_argument(
            '--name', type=str, default='debug', help='Experiment name prefix.')
        self.parser.add_argument(
            '--seed', type=int, default=0, help='Random Seed.')
        self.parser.add_argument(
            '--batch_size', type=int, default=1, help='Batch size.')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1',
                                 help='Comma-separated list of GPU IDs.')
        self.parser.add_argument(
            '--num_workers', default=8, type=int, help='Number of threads for the DataLoader.')
        self.parser.add_argument('--init_method', type=str, default='kaiming', choices=(
            'kaiming', 'normal', 'xavier'), help='Initialization method to use for conv kernels and linear weights.')
        self.parser.add_argument(
            '--data_dir', type=str, default='/data1/datasets/gutgan', help='Directory for data.')
        self.parser.add_argument(
            '--save_dir', type=str, default='/data1/pix2pixHD', help='Directory for results \
                including ckpts and viz (prefix).')
        self.parser.add_argument('--maximize_metric', default=False, action="store_true",
                                 help='For evaluation, higher the metric the better, else lower.')

        # Data Args
        self.parser.add_argument('--input_channels', type=int, default=3,
                                 help='Number of input channels of the generator.')
        self.parser.add_argument('--output_channels', type=int, default=3,
                                 help='Number of output channels of the generator.')
        self.parser.add_argument('--num_red_threshold', type=float, default=1000,
                                 help='Filter patches with num red below this threshold.')
        self.parser.add_argument("--synthetic_data_dir", type=str, default=None,
                                 help='Directory holding synthetic images and csv file.')
        self.parser.add_argument("--synthetic_data_proportion", type=float,
                                 default=0, help='Proportion of training images which are synthetic.')

        # Logger Args
        self.parser.add_argument('--steps_per_print', type=int, default=100,
                                 help='Number of steps between printing loss to the console and TensorBoard.')
        self.parser.add_argument('--num_visuals', type=int, default=10,
                                 help='Number of images to visualize.')
        self.parser.add_argument('--steps_per_visual', type=int, default=200,
                                 help='Number of steps between visualizing training examples.')

        self.parser.add_argument(
            '--max_eval', type=int, default=None, help='Max data points to evaluate on.')
        self.parser.add_argument(
            '--start_epoch', type=int, default=1, help='Epoch to start training')
        self.parser.add_argument('--load_epoch', type=int, default=0,
                                 help='Default uses latest cached model if continue train or eval set')

        # Saver Args
        self.parser.add_argument('--ckpt_path', type=str, default=None,
                                 help='Path to model checkpoint')

    def parse_args(self):
        """
        Function that parses arguments, create checkpoints and vizualization directory, and sets up gpu device.

        Returns
        -------
        args : Namespace
            Parsed program arguments
        """
        args = self.parser.parse_args()

        # Limit sources of nondeterministic behavior
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

        # Causes cuDNN to deterministically select an algorithm
        # possibly at the cost of reduced performance
        if args.debug:
            torch.backends.cudnn.benchmark = False

        if hasattr(self, 'isTrain'):
            args.isTrain = self.isTrain   # train or test

        if args.isTrain and not args.continue_train:
            args.name = datetime.now().strftime('%y%m%d_%H%M%S') + '_' + args.name

        os.makedirs(os.path.join(args.save_dir, args.name),  exist_ok=True)

        # Save args to a JSON file
        prefix = 'train' if args.isTrain else 'test'
        with open(os.path.join(args.save_dir, args.name, f"{prefix}_args.json"), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')

        # Create ckpt dir and viz dir
        args.ckpt_dir = os.path.join(args.save_dir, args.name, 'ckpts')
        os.makedirs(args.ckpt_dir, exist_ok=True)

        args.viz_dir = os.path.join(args.save_dir, args.name, 'viz')
        os.makedirs(args.viz_dir, exist_ok=True)

        # Set up available GPUs
        def args_to_list(csv, arg_type=int):
            """Convert comma-separated arguments to a list."""
            arg_vals = [arg_type(d) for d in str(csv).split(',')]
            return arg_vals

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        args.gpu_ids = args_to_list(args.gpu_ids)

        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(0)
            args.gpu_ids = ['cuda' + ':' + str(gpu_id)
                            for gpu_id in args.gpu_ids]
            args.device = 'cuda'
        else:
            args.device = 'cpu'

        if hasattr(args, 'supervised_factors'):
            args.supervised_factors = args_to_list(args.supervised_factors)

        # Ensure consistency of load_epoch and start_epoch arguments with each other and defaults.
        if not args.isTrain or args.continue_train:
            if args.load_epoch > 0:
                args.start_epoch = args.load_epoch + 1
            elif args.start_epoch > 1:
                args.load_epoch = args.start_epoch - 1
            else:
                args.load_epoch = self.get_last_saved_epoch(args)
                args.start_epoch = args.load_epoch + 1

        self.print_options(args)

        return args

    def get_last_saved_epoch(self, args):
        """Returns the last epoch at which a checkpoint was saved.

        Parameters
        ----------
        args : Namespace
            Arguments for models and model testing

        Returns
        -------
        epoch : int
            Last epoch at which checkpoints were saved
        """
        ckpt_files = sorted([name for name in os.listdir(
            args.ckpt_dir) if name.split(".", 1)[1] == "pth.tar"])

        if 'best.pth.tar' in ckpt_files:
            ckpt_files.remove('best.pth.tar')

        if len(ckpt_files) > 0:
            epoch = int(ckpt_files[-1][:3])
        else:
            epoch = 0
        return epoch

    def print_options(self, args):
        """
        Function that prints and save options
        It will print both current options and default values(if different).
        Inspired by:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py#L88-L111

        Parameters
        ----------
        args : Namespace
            Arguments for models and model testing
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)