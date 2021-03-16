"""
Defines the dataset class and associated functions of the asr pipeline.
Adapted from https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio

from librosa.filters import mel as librosa_mel_fn

class Audio2Mel(nn.Module):
    """
    https://github.com/descriptinc/melgan-neurips/blob/6488045bfba1975602288de07a58570c7b4d66ea/mel2wav/modules.py#L26-L69
    """
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = audio.unsqueeze(1)
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec

class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

            # torchaudio.transforms.MelSpectrogram(
            #     sample_rate=sample_rate, n_mels=128),

def get_audio_transforms(phase, spec, sample_rate=22050):
    audio_2_mel = Audio2Mel()
    audio_2_mel_transform = audio_2_mel.forward
    if phase == 'train':
        transforms = []
        if not spec:
            transforms.append(torchvision.transforms.Lambda(lambd=lambda x: audio_2_mel(x)))

        transforms += [
            torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
            torchaudio.transforms.TimeMasking(time_mask_param=100)
        ]
        transforms = torchvision.transforms.Compose(transforms)

    elif phase == 'valid':
        # transforms = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=sample_rate, n_mels=128
        # )
        transforms = audio_2_mel_transform

    return transforms


def data_processing(data, phase, text_transform):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    # for (waveform, sample_rate, utterance, _, _, _) in data:
    for (data, sample_rate, utterance, speaker_id, _, spec) in data:
        # audio_transforms = get_audio_transforms(phase, sample_rate)
        audio_transforms = get_audio_transforms(phase, spec)
        spec = audio_transforms(data).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(
        spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    # print(spectrograms, labels, input_lengths, label_lengths)
    return spectrograms, labels, input_lengths, label_lengths
