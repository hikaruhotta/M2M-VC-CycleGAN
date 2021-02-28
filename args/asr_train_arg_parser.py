"""
Arguments for training ASR model.
Inherits BaseArgParser.
"""

from args.base_arg_parser import BaseArgParser


class ASRTrainArgParser(BaseArgParser):
    """
    Class which implements an argument parser for args used only in train mode.
    It inherits BaseArgParser.
    """

    def __init__(self):
        super(ASRTrainArgParser, self).__init__()
        self.isTrain = True

        self.parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train.')
        self.parser.add_argument(
            '--dropout', type=float, default=0.1, help='Dropout rate.')
        self.parser.add_argument(
            '--lr', type=float, default=5e-4, help='Learning rate.')
        self.parser.add_argument(
            '--gamma', type=float, default=0.99, help='Annealing rate for LR scheduler.')

        # Model args
        self.parser.add_argument(
            '--n_cnn_layers', type=int, default=3, help='Numer of CNN layers.')
        self.parser.add_argument(
            '--n_rnn_layers', type=int, default=5, help='Number of RNN layers')
        self.parser.add_argument(
            '--rnn_dim', type=int, default=512, help='Dimensionality of RNN')
        self.parser.add_argument(
            '--n_class', type=int, default=29, help='Number of output classes.')
        self.parser.add_argument(
            '--n_feats', type=int, default=128, help='Number of features.')
        self.parser.add_argument(
            '--stride', type=int, default=2, help='Conv2D kernel stride.')

        self.parser.add_argument('--max_ckpts', type=int, default=3, help='Max ckpts to save.')
        self.parser.add_argument('--epochs_per_save', type=int, default=1,
                                 help='Number of epochs between saving a checkpoint to save_dir.')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
