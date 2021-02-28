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
            '--target_id', type=str, default="PRV_se0_ag1_f_02_2", help='Target speaker id.')

        # Model args
