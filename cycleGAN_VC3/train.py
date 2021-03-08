import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data
import pickle

from cycleGAN_VC3.model import Generator, Discriminator
from args.cycleGAN_train_arg_parser import CycleGANTrainArgParser
from dataset.dataset import Dataset
from dataset.vc_dataset import trainingDataset
from cycleGAN_VC3.utils import get_audio_transforms, data_processing, decode_melspectrogram, get_img_from_fig, get_waveform_fig
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver


class CycleGANTraining(object):
    def __init__(self, args):
        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.generator_lr = args.generator_lr
        self.discriminator_lr = args.discriminator_lr
        self.decay_after = args.decay_after
        self.generator_lr_decay = self.generator_lr / self.num_epochs
        self.discriminator_lr_decay = self.discriminator_lr / self.num_epochs
        self.mini_batch_size = args.batch_size
        self.cycle_loss_lambda = args.cycle_loss_lambda
        self.identity_loss_lambda = args.identity_loss_lambda
        self.device = args.device
        
        self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
        self.sample_rate = args.sample_rate

        self.dataset_A = self.loadPickleFile(args.normalized_dataset_A_path)
        dataset_A_norm_stats = np.load(args.norm_stats_A_path)
        self.dataset_A_mean = dataset_A_norm_stats['mean_A']  # fix to mean and std after running data preprocessing script again
        self.dataset_A_std = dataset_A_norm_stats['std_A']
        self.dataset_B = self.loadPickleFile(args.normalized_dataset_B_path)
        dataset_B_norm_stats = np.load(args.norm_stats_B_path)
        self.dataset_B_mean = dataset_B_norm_stats['mean']
        self.dataset_B_std = dataset_B_norm_stats['std']

        self.n_samples = len(self.dataset_A)
        self.dataset = trainingDataset(datasetA=self.dataset_A,
                                    datasetB=self.dataset_B,
                                    n_frames=64)
        self.train_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                    batch_size=self.mini_batch_size,
                                                    shuffle=True,
                                                    drop_last=False)
        
        self.logger = TrainLogger(args, len(self.train_dataloader.dataset))
        self.saver = ModelSaver(args)

        # Generator and Discriminator
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)

        # Optimizer
        g_params = list(self.generator_A2B.parameters()) + \
            list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
            list(self.discriminator_B.parameters())

        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

        # Storing Discriminatior and Generator Loss

    def adjust_lr_rate(self, optimizer, name='generator'):
        if name == 'generator':
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0., self.discriminator_lr - self.discriminator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.discriminator_lr

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def loadPickleFile(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.start_epoch()
            
            for i, (real_A, real_B) in enumerate(tqdm(self.train_dataloader)):
                self.logger.start_iter()
                num_iterations = (
                    self.n_samples // self.mini_batch_size) * epoch + i
                
                real_A = real_A.to(self.device, dtype=torch.float)
                real_B = real_B.to(self.device, dtype=torch.float)

                # Train Generator
                fake_B = self.generator_A2B(real_A)
                cycle_A = self.generator_B2A(fake_B)
                fake_A = self.generator_B2A(real_B)
                cycle_B = self.generator_A2B(fake_A)
                identity_A = self.generator_B2A(real_A)
                identity_B = self.generator_A2B(real_B)
                d_fake_A = self.discriminator_A(fake_A)
                d_fake_B = self.discriminator_B(fake_B)

                # Generator Cycle Loss
                cycleLoss = torch.mean(
                    torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

                # Generator Identity Loss
                identityLoss = torch.mean(
                    torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

                # Generator Loss
                g_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                g_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

                # Total Generator Loss
                g_loss = g_loss_A2B + g_loss_B2A + \
                    self.cycle_loss_lambda * cycleLoss + self.identity_loss_lambda * identityLoss
                # self.generator_loss_store.append(generator_loss.item())

                # Backprop for Generator
                self.reset_grad()
                g_loss.backward()
                self.generator_optimizer.step()

                # Train Discriminator

                # Discriminator Feed Forward
                d_real_A = self.discriminator_A(real_A)
                d_real_B = self.discriminator_B(real_B)

                generated_A = self.generator_B2A(real_B)
                d_fake_A = self.discriminator_A(generated_A)

                # For Second Step Adverserial Loss A->B
                cycled_B = self.generator_A2B(generated_A)
                d_cycled_B = self.discriminator_B(cycled_B)

                generated_B = self.generator_A2B(real_A)
                d_fake_B = self.discriminator_B(generated_B)

                # For Second Step Adverserial Loss B->A
                cycled_A = self.generator_B2A(generated_B)
                d_cycled_A = self.discriminator_A(cycled_A)

                # Loss Functions
                d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
                d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
                d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

                d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
                d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                # Second Step Adverserial Loss
                d_loss_A_cycled = torch.mean((0 - d_cycled_A) ** 2)
                d_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)
                d_loss_A_2nd = (d_loss_A_real + d_loss_A_cycled) / 2.0
                d_loss_B_2nd = (d_loss_B_real + d_loss_B_cycled) / 2.0

                # Final Loss for discriminator with the second step adverserial loss
                d_loss = (d_loss_A + d_loss_B) / 2.0 + \
                    (d_loss_A_2nd + d_loss_B_2nd) / 2.0
                # self.discriminator_loss_store.append(d_loss.item())

                # Backprop for Discriminator
                self.reset_grad()
                d_loss.backward()
                self.discriminator_optimizer.step()

                # if num_iterations % args.steps_per_print == 0:
                #     print(f"Epoch: {epoch} Step: {num_iterations} Generator Loss: {generator_loss.item()} Discriminator Loss: {d_loss.item()}")
                
                self.logger.log_iter(loss_dict={'g_loss': g_loss.item(), 'd_loss': d_loss.item()})
                
                self.logger.end_iter()
                # adjust learning rates
                if self.logger.global_step > self.decay_after:
                    identity_loss_lambda = 0
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='generator')
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='discriminator')
            
            wav_A = decode_melspectrogram(self.vocoder, generated_A[0].detach().cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
            wav_A_fig = get_waveform_fig(wav_A, self.sample_rate)
            wav_B = decode_melspectrogram(self.vocoder, generated_B[0].detach().cpu().numpy(), self.dataset_B_mean, self.dataset_B_std)
            self.logger.log_img(wav_A_fig, "debug")
            # generated_A.shape, generated_B.shape
            
            self.logger.end_epoch()


if __name__ == "__main__":
    parser = CycleGANTrainArgParser()
    args = parser.parse_args()
    cycleGAN = CycleGANTraining(args)
    cycleGAN.train()
