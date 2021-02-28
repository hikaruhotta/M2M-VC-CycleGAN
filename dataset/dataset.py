"""
Defines Dataset which is a wrapper for the torch.utils.data.Dataset class.
"""

import pandas as pd
from pathlib import Path

import torch
import torch.utils.data as data
import torchaudio


class Dataset(data.Dataset):

    def __init__(self, args, split='train', coraal=True, voc=False):
        """
        Args:
            args (Namespace): Program arguments
            split (str): 'train', 'validation', or 'test' set
            coraal (bool): Return CORAAL samples
            voc (bool): Return VOC samples
        """
        self.split = split

        # Load manifest file which defines dataset
        data_base_dir = Path(args.data_dir)
        coraal_manifest_path = "./manifests/coraal_manifest.csv"
        self.df = pd.read_csv(coraal_manifest_path, sep=',')

        # Filter samples in split (train/val/test)
        self.df = self.df[self.df['split'] == split]

        self.wav_files = self.df['wav_file'].tolist()
        self.txt_files = self.df['txt_file'].tolist()

        # Contruct wav and txt paths
        # TO-DO: Change these to relative paths once manifests are updated
        self.wav_paths = self.wav_files
        self.txt_paths = self.txt_files
        # self.wav_paths = [data_base_dir / self.wav_files[i] for i in range(len(self.wav_files))]
        # self.txt_paths = [data_base_dir / self.txt_files[i] for i in range(len(self.txt_files))]

        self.ground_truth_text = self.df['groundtruth_text_train'].tolist()
        self.durations = self.df['duration'].tolist()
        self.speaker_ids = self.df['speaker_id'].tolist()
        self.genders = self.df['gender'].tolist()

        # Sanity check that txt_path contents are the same as ground_truth_text
        # TO-DO: Remove this once sanity check is confirmed
        # characters = {}
        # for i, (txt_path, gt) in enumerate(zip(self.txt_paths, self.ground_truth_text)):
        #     with open(txt_path, 'r') as reader:
        #         # assert reader.read() == gt
        #         if len(characters):
        #             characters |= set(gt.lower())
        #         else:
        #             characters = set(gt.lower())
        
        # characters = list(characters)
        # characters.sort()
        # print(characters)



    def __getitem__(self, index):
        """
        Loads audio file and labels into a tuple based on index.
        Args:
            index (int): Index of sample to return
        Returns:
            item (tuple): tuple of audio and labels
        """
        items = []

        if hasattr(self, 'wav_paths'):
            waveform, sample_rate = torchaudio.load(self.wav_paths[index])
            items = [waveform, sample_rate]
        if hasattr(self, 'ground_truth_text'):
            items.append(self.ground_truth_text[index])
        if hasattr(self, 'speaker_ids'):
            items.append(self.speaker_ids[index])
        if hasattr(self, 'genders'):
            items.append(self.genders[index])
        
        # Returns (waveform, sample_rate, ground_truth_text, speaker_ids, gender)
        return tuple(items)

    def __len__(self):
        """
        Gets the length of the dataset.
        Returns:
            int: The length of the dataset
        """
        return len(self.df)