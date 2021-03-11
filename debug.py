
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio

from librosa.filters import mel as librosa_mel_fn

from asr.data import Audio2Mel

path = "/home/ubuntu/data/datasets/data_processed_voc/wav/voc_26_part_65.wav"

vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')


attr = ['n_fft',
'hop_length',
'win_length',
'sampling_rate',
'n_mel_channels']
for att in attr:
    print(att, getattr(vocoder.fft, att))
audio_2_mel = Audio2Mel()

torch.manual_seed(1)

for i in range(10, 15):
    path = f"/home/ubuntu/data/datasets/data_processed_voc/wav/voc_0_part_{i}.wav"
    wav, _ = torchaudio.load(path)
    spec1 = audio_2_mel.forward(wav)
    spec2 = vocoder(wav)
    diff = torch.mean(spec1.cpu() - spec2.cpu())
    print(diff)
