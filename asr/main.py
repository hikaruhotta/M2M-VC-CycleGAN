"""
Main function of the asr pipeline.
Adapted from https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO
"""

import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchaudio
from asr.data import TextTransform, get_audio_transforms, data_processing
from asr.models import SpeechRecognitionModel
from asr.train import train
from asr.test import test
from asr.utils import IterMeter
from logger.train_logger import TrainLogger
from args.asr_train_arg_parser import ASRTrainArgParser


def main(args, train_url="train-clean-100", test_url="test-clean"):
    train_dataset = torchaudio.datasets.LIBRISPEECH(
        args.data_dir, url=train_url, download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(
        args.data_dir, url=test_url, download=True)

    train_audio_transforms = get_audio_transforms('train')
    valid_audio_transforms = get_audio_transforms('valid')

    text_transform = TextTransform()

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   collate_fn=lambda x: data_processing(
                                       x, train_audio_transforms, text_transform),
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(
                                      x, valid_audio_transforms, text_transform),
                                  num_workers=args.num_workers,
                                  pin_memory=True)

    model = SpeechRecognitionModel(
        args.n_cnn_layers, args.n_rnn_layers, args.rnn_dim,
        args.n_class, args.n_feats, args.stride, args.dropout
    ).to(args.device)

    # print(model)
    print('Num Model Parameters', sum(
        [param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), args.lr)
    criterion = nn.CTCLoss(blank=28).to(args.device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                              steps_per_epoch=int(
                                                  len(train_loader)),
                                              epochs=args.num_epochs,
                                              anneal_strategy='linear')

    logger = TrainLogger(args, len(train_loader.dataset))
    logger.log_hparams(args)

    iter_meter = IterMeter()
    for epoch in range(1, args.num_epochs + 1):
        train(args, model, train_loader, criterion, optimizer, scheduler, logger)
        test(args, model, test_loader, criterion, logger)


if __name__ == "__main__":
    parser = ASRTrainArgParser()
    args = parser.parse_args()
    libri_train_set = "train-clean-100"
    libri_test_set = "test-clean"
    main(args, libri_train_set, libri_test_set)
