"""
Defines the util functions associated with the cycleGAN VC pipeline.
"""

import torch
import torch.nn as nn
import torchaudio

def get_audio_transforms(phase, sample_rate=16000):
    if phase == 'train':
        transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_mels=144),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
            torchaudio.transforms.TimeMasking(time_mask_param=100)
        )
    elif phase == 'valid':
        transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=144
        )

    return transforms


def data_processing(data, phase):
    spectrograms_A = []
    spectrograms_B = []

    for (input_A, input_B) in data:
        (waveform_A, sample_rate_A, _, _, _) = input_A
        audio_transforms_A = get_audio_transforms(phase, sample_rate_A)
        spec_A = audio_transforms_A(waveform_A).squeeze(0).transpose(0, 1)[:100,:]
        print(f'spec_A shape is {spec_A.shape}')
        spectrograms_A.append(spec_A)

        (waveform_B, sample_rate_B, _, _, _) = input_B
        audio_transforms_B = get_audio_transforms(phase, sample_rate_B)
        spec_B = audio_transforms_B(waveform_B).squeeze(0).transpose(0, 1)[:100,:]
        print(f'spec_B shape is {spec_B.shape}')
        spectrograms_B.append(spec_B)

    spectrograms_A = nn.utils.rnn.pad_sequence(
        spectrograms_A, batch_first=True).transpose(1, 2)
    spectrograms_B = nn.utils.rnn.pad_sequence(
        spectrograms_B, batch_first=True).transpose(1, 2)
    
    return (spectrograms_A, spectrograms_B)