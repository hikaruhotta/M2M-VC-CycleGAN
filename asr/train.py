"""
Train function of the asr pipeline.
Adapted from https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO
"""

import torch.nn.functional as F
from tqdm import tqdm
from asr.utils import save_ckpt


def train(args, model, train_loader, criterion, optimizer, scheduler, logger):
    model.train()
    data_len = len(train_loader.dataset)
    logger.start_epoch()
    for batch_idx, _data in enumerate(tqdm(train_loader)):
        logger.start_iter()

        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(args.device), labels.to(args.device)
        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        logger.log_iter(loss_dict={'train_loss': loss.item()})

        optimizer.step()
        scheduler.step()

        logger.end_iter()
