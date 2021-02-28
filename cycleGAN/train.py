import os
import numpy as np
import argparse
import time
import librosa
import pickle

from cycleGAN.model import Generator, Discriminator
from tqdm import tqdm

import torch
import torch.utils.data as data


class CycleGANTraining(object):
    def __init__(self, args):
        self.generator_lr = args.generator_lr
        self.discriminator_lr = args.discriminator_lr
        self.decay_after = args.decay_after
        self.generator_lr_decay = self.generator_lr / self.num_epochs
        self.discriminator_lr_decay = self.discriminator_lr / self.num_epochs

        self.cycle_loss_lambda = args.cycle_loss_lambda
        self.identity_loss_lambda = args.identity_loss_lambda

        self.train_dataset = None  # TODO: replace with Sofian's
        self.train_dataloader = data.DataLoader(dataset=self.train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

        # Generator and Discriminator
        self.generator_A2B = Generator().to(args.device)
        self.generator_B2A = Generator().to(args.device)
        self.discriminator_A = Discriminator().to(args.device)
        self.discriminator_B = Discriminator().to(args.device)

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
        self.generator_loss_store = []
        self.discriminator_loss_store = []

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

    def train(self):
        # Training Begins
        for epoch in range(self.start_epoch, self.num_epochs):
            # Preparing Dataset
            n_samples = len(self.dataset)

            for i, (real_A, real_B) in enumerate(tqdm(train_loader)):
                num_iterations = (n_samples // self.mini_batch_size) * epoch + i
                    
                if num_iterations > self.decay_after:
                    identity_loss_lambda = 0
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='generator')
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='discriminator')

                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)

                # Train Generator

                fake_B = self.generator_A2B(real_A)
                cycle_A = self.generator_B2A(fake_B)

                fake_A = self.generator_B2A(real_B)
                cycle_B = self.generator_A2B(fake_A)

                identity_A = self.generator_B2A(real_A)
                identity_B = self.generator_A2B(real_B)

                d_fake_A = self.discriminator_A(fake_A)
                d_fake_B = self.discriminator_B(fake_B)

                # For Second Step Adverserial Loss
                d_fake_cycle_A = self.discriminator_A(cycle_A)  # TODO: comment out since not being used???
                d_fake_cycle_B = self.discriminator_B(cycle_B)

                # Generator Cycle Loss
                cycleLoss = torch.mean(
                    torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

                # Generator Identity Loss
                identiyLoss = torch.mean(
                    torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

                # Generator Loss
                generator_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                generator_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

                # Total Generator Loss
                generator_loss = generator_loss_A2B + generator_loss_B2A + \
                    cycle_loss_lambda * cycleLoss + identity_loss_lambda * identiyLoss
                self.generator_loss_store.append(generator_loss.item())

                # Backprop for Generator
                self.reset_grad()
                generator_loss.backward()
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
                self.discriminator_loss_store.append(d_loss.item())

                # Backprop for Discriminator
                self.reset_grad()
                d_loss.backward()
                self.discriminator_optimizer.step()
                
                if num_iterations % args.steps_per_print == 0:
                    print(f"Epoch: {epoch} Step: {num_iterations} Generator Loss: {generator_loss.item()} Discriminator Loss: {d_loss.item()}"

            if epoch % 2000 == 0 and epoch != 0:
                print(f"Epoch: {epoch} Generator Loss: {generator_loss.item()} Discriminator Loss: {d_loss.item()}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train CycleGAN using source dataset and target dataset")

    logf0s_normalization_default = './cache/logf0s_normalization.npz'
    mcep_normalization_default = './cache/mcep_normalization.npz'
    coded_sps_A_norm = './cache/coded_sps_A_norm.pickle'
    coded_sps_B_norm = './cache/coded_sps_B_norm.pickle'
    model_checkpoint = './model_checkpoint/'
    resume_training_at = './model_checkpoint/_CycleGAN_CheckPoint'
    #     resume_training_at = None

    validation_A_dir_default = './data/S0913/'
    output_A_dir_default = './converted_sound/S0913'

    validation_B_dir_default = './data/gaoxiaosong/'
    output_B_dir_default = './converted_sound/gaoxiaosong/'

    parser.add_argument('--logf0s_normalization', type=str,
                        help="Cached location for log f0s normalized", default=logf0s_normalization_default)
    parser.add_argument('--mcep_normalization', type=str,
                        help="Cached location for mcep normalization", default=mcep_normalization_default)
    parser.add_argument('--coded_sps_A_norm', type=str,
                        help="mcep norm for data A", default=coded_sps_A_norm)
    parser.add_argument('--coded_sps_B_norm', type=str,
                        help="mcep norm for data B", default=coded_sps_B_norm)
    parser.add_argument('--model_checkpoint', type=str,
                        help="location where you want to save the model", default=model_checkpoint)
    parser.add_argument('--resume_training_at', type=str,
                        help="Location of the pre-trained model to resume training",
                        default=resume_training_at)
    parser.add_argument('--validation_A_dir', type=str,
                        help="validation set for sound source A", default=validation_A_dir_default)
    parser.add_argument('--output_A_dir', type=str,
                        help="output for converted Sound Source A", default=output_A_dir_default)
    parser.add_argument('--validation_B_dir', type=str,
                        help="Validation set for sound source B", default=validation_B_dir_default)
    parser.add_argument('--output_B_dir', type=str,
                        help="Output for converted sound Source B", default=output_B_dir_default)

    argv = parser.parse_args()

    logf0s_normalization = argv.logf0s_normalization
    mcep_normalization = argv.mcep_normalization
    coded_sps_A_norm = argv.coded_sps_A_norm
    coded_sps_B_norm = argv.coded_sps_B_norm
    model_checkpoint = argv.model_checkpoint
    resume_training_at = argv.resume_training_at

    validation_A_dir = argv.validation_A_dir
    output_A_dir = argv.output_A_dir
    validation_B_dir = argv.validation_B_dir
    output_B_dir = argv.output_B_dir

    # Check whether following cached files exists
    if not os.path.exists(logf0s_normalization) or not os.path.exists(mcep_normalization):
        print(
            "Cached files do not exist, please run the program preprocess_training.py first")

    cycleGAN = CycleGANTraining(logf0s_normalization=logf0s_normalization,
                                mcep_normalization=mcep_normalization,
                                coded_sps_A_norm=coded_sps_A_norm,
                                coded_sps_B_norm=coded_sps_B_norm,
                                model_checkpoint=model_checkpoint,
                                validation_A_dir=validation_A_dir,
                                output_A_dir=output_A_dir,
                                validation_B_dir=validation_B_dir,
                                output_B_dir=output_B_dir,
                                restart_training_at=resume_training_at)
    cycleGAN.train()
