"""
Train function of the asr pipeline.
Adapted from https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO
"""

import torch.nn.functional as F
from tqdm import tqdm
from asr.utils import save_ckpt


def train(args, model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, logger):
    model.train()
    data_len = len(train_loader.dataset)
    logger.start_epoch()
    for batch_idx, _data in tqdm(enumerate(train_loader)):
        logger.start_iter()

        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        logger.log_iter(loss_dict={'train_loss': loss.item()})
        logger.log_metrics({'learning_rate': scheduler.get_lr()})

        optimizer.step()
        scheduler.step()
        iter_meter.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item()))

        logger.end_iter()

    if logger.epoch % args.epochs_per_save == 0:
        save_ckpt(logger.epoch, model, "SpeechRecognitionModel",
                  optimizer, scheduler, args.ckpt_dir, args.device)

    logger.end_epoch()
