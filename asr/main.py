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
from saver.model_saver import ModelSaver
from dataset.dataset import Dataset


def main(args):
    if args.librispeech:
        print("Loading Librispeech dataset!")
        train_dataset = torchaudio.datasets.LIBRISPEECH(
            args.data_dir, url="train-clean-360", download=True)
        valid_dataset = torchaudio.datasets.LIBRISPEECH(
            args.data_dir, url="test-clean", download=True)
    else:
        train_dataset = Dataset(args, "train", return_pair=args.return_pair)
        valid_dataset = Dataset(args, "val", return_pair=args.return_pair)

    print(f"Training set has {len(train_dataset)} samples. Validation set has {len(valid_dataset)} samples.")
        # train_audio_transforms = get_audio_transforms('train')
        # valid_audio_transforms = get_audio_transforms('valid')

    text_transform = TextTransform()

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   collate_fn=lambda x: data_processing(
                                       x, "train", text_transform),
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   collate_fn=lambda x: data_processing(
                                       x, "valid", text_transform),
                                   num_workers=args.num_workers,
                                   pin_memory=True)

    model = SpeechRecognitionModel(
        args.n_cnn_layers, args.n_rnn_layers, args.rnn_dim,
        args.n_class, args.n_feats, args.stride, args.dropout
    ).to(args.device)

    print('Num Model Parameters', sum(
        [param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), args.lr)
    criterion = nn.CTCLoss(blank=28).to(args.device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                              steps_per_epoch=int(
                                                  len(train_loader)),
                                              epochs=args.num_epochs,
                                              anneal_strategy='linear')
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)

    saver = ModelSaver(args, max_ckpts=args.max_ckpts,
                       metric_name="test_wer", maximize_metric=False)

    if args.continue_train:
        saver.load_model(model, "SpeechRecognitionModel",
                         args.ckpt_path, optimizer, scheduler)
    elif args.pretrained_ckpt_path:
        saver.load_model(model, "SpeechRecognitionModel",
                         args.pretrained_ckpt_path, None, None)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    logger = TrainLogger(args, len(train_loader.dataset))
    logger.log_hparams(args)

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        train(args, model, train_loader, criterion,
              optimizer, scheduler, logger)
        if logger.epoch % args.epochs_per_save == 0:
            metric_dict = test(args, model, valid_loader, criterion, logger)
            saver.save(logger.epoch, model, optimizer, scheduler, args.device,
                       "SpeechRecognitionModel", metric_dict["test_wer"])
        logger.end_epoch()


if __name__ == "__main__":
    parser = ASRTrainArgParser()
    args = parser.parse_args()
    main(args)
