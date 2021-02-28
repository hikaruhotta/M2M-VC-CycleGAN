"""
Arguments for CycleGAN VC.
Inherits BaseArgParser.
"""

from args.train_arg_parser import TrainArgParser


class CycleGANTrainArgParser(TrainArgParser):
    """
    Class which implements an argument parser for args used only in training CycleGAN VC.
    It inherits BaseArgParser.
    """

    def __init__(self):
        super(CycleGANTrainArgParser, self).__init__()

        self.parser.add_argument(
            '--source_id', type=str, default="ATL_se0_ag1_f_03_1", help='Source speaker id.')
        self.parser.add_argument(
            '--target_id', type=str, default="28", help='Target speaker id.')

        # Model args
        self.parser.add_argument(
            '--generator_lr', type=float, default=2e-4, help='Initial generator learning rate.')
        self.parser.add_argument(
            '--discriminator_lr', type=float, default=1e-4, help='Initial discrminator learning rate.')
        
        # Loss lambdas
        self.parser.add_argument(
            '--cycle_loss_lambda', type=float, default=10, help='Lambda value for cycle consistency loss.')
        self.parser.add_argument(
            '--cycle_loss_lambda', type=float, default=5, help='Lambda value for identity loss.')
        
        self.parser.set_defaults(num_epochs=200000, decay_after=10000, start_epoch=1, steps_per_print=100, epochs_per_save=5, 
                                 steps_per_visual=1000, gaussian_kernel_size=51)