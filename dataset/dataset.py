"""
Defines Dataset which is a wrapper for the torch.utils.data.Dataset class.
"""

import pandas as pd
from pathlib import Path

import torch
import torch.utils.data as data
import torchaudio
import random

class Dataset(data.Dataset):

    def __init__(self, args, split='train', coraal=True, voc=False, return_pair=False):
        """
        Args:
            args (Namespace): Program arguments
            split (str): 'train', 'validation', or 'test' set
            coraal (bool): Return CORAAL samples
            voc (bool): Return VOC samples
            return_pair (bool): Return a pair of CORAAL and VOC samples (For training CycleGAN)
        """
        self.split = split
        self.coraal = coraal
        self.voc = voc
        if self.split == 'val':
            self.coraal = True
            self.voc = False
        self.return_pair = return_pair
        self.datasets = ['coraal']*self.coraal + ['voc']*self.voc
        self.base_dir = Path(args.data_dir)
        self.manifest_path = Path(args.manifest_path)

        # Sanity check: return_pair is only valid if using both datasets
        if self.return_pair:
            assert self.coraal and self.voc
            self.coraal_df, self.coraal_wav_paths, self.coraal_txt_paths, self.coraal_ground_truth_text, self.coraal_durations, self.coraal_speaker_ids = self._read_manifest(self.base_dir, dataset="coraal", speaker_id=args.target_id)
            self.voc_df, self.voc_wav_paths, self.voc_txt_paths, self.voc_ground_truth_text, self.voc_durations, self.voc_speaker_ids = self._read_manifest(self.base_dir, dataset="voc", speaker_id=args.source_id)
        else:
            # Merge dataframes
            self.df = None
            if self.coraal:
                if args.small_dataset:
                    self.df = pd.read_csv(self.manifest_path / "coraal_small_manifest.csv", sep=',')
                else:
                    self.df = pd.read_csv(self.manifest_path / "coraal_manifest.csv", sep=',')
            if self.voc:
                self.df = pd.read_csv(self.manifest_path / "voc_manifest.csv", sep=',').append(self.df, ignore_index=True)
                
            self.df, self.wav_paths, self.txt_paths, self.ground_truth_text, self.durations, self.speaker_ids = self._read_manifest(self.base_dir, split=split, df=self.df)



        # self.genders = self.df['gender'].tolist()

        # Sanity check that txt_path contents are the same as ground_truth_text
        # TO-DO: Remove this once sanity check is confirmed
        # characters = {}
        # for i, (txt_path, gt) in enumerate(zip(self.txt_paths, self.ground_truth_text)):
        #     with open(txt_path, 'r') as reader:
        #         # assert reader.read() == gt
        #         if '-' in gt:
        #             print(gt)
        #         if len(characters):
        #             characters |= set(gt.lower())
        #         else:
        #             characters = set(gt.lower())
        
        # characters = list(characters)
        # characters.sort()
        # print(characters)


    def _read_manifest(self, data_dir, split=None, dataset=None, df=None, speaker_id=None):

        if df is None:
            # Load manifest file which defines dataset
            manifest_file = self.manifest_path / f"{dataset}_manifest.csv"
            df = pd.read_csv(manifest_file, sep=',')

        if speaker_id is not None:
            # If done voice conversion, then filter by speaker_id
            df['speaker_id'] = df['speaker_id'].astype(str)
            df = df[df['speaker_id'] == speaker_id]
        else:
            # Filter samples in split (train/val/test)
            df = df[df['split'] == split]

        # print(f'dataset {dataset} df has {len(df)} elements')
        wav_files = df['wav_file'].tolist()
        txt_files = df['txt_file'].tolist()

        # Contruct wav and txt paths
        # TO-DO: Change these to relative paths once manifests are updated
        wav_paths = [data_dir / wav_files[i] for i in range(len(wav_files))]
        txt_paths = [data_dir / txt_files[i] for i in range(len(txt_files))]

        ground_truth_text = df['groundtruth_text_train'].tolist()
        durations = df['duration'].tolist()
        speaker_ids = df['speaker_id'].tolist()

        return df, wav_paths, txt_paths, ground_truth_text, durations, speaker_ids

    def __getitem__(self, index):
        """
        Loads audio file and labels into a tuple based on index.
        Args:
            index (int): Index of sample to return
        Returns:
            item (tuple): tuple of audio and labels
        """
        items = []

        if self.return_pair:
            coraal_index = random.randint(0, len(self.coraal_df) - 1)
            waveform_A, sample_rate_A = torchaudio.load(self.coraal_wav_paths[coraal_index])
            items.append((waveform_A, sample_rate_A, self.coraal_ground_truth_text[coraal_index], self.coraal_speaker_ids[coraal_index], self.coraal_durations[coraal_index], None))
            voc_index = random.randint(0, len(self.voc_df) - 1)
            waveform_B, sample_rate_B = torchaudio.load(self.voc_wav_paths[voc_index])
            items.append((waveform_B, sample_rate_B, self.voc_ground_truth_text[voc_index], self.voc_speaker_ids[voc_index], self.voc_durations[voc_index], None))

        else:
            waveform, sample_rate = torchaudio.load(self.wav_paths[index])
            items = [waveform, sample_rate, self.ground_truth_text[index], self.speaker_ids[index], self.durations[index], None]
        
        # Returns (waveform, sample_rate, ground_truth_text, speaker_ids, duration)
        return tuple(items)

    def __len__(self):
        """
        Gets the length of the dataset.
        Returns:
            int: The length of the dataset
        """
        if self.return_pair:
            return max(len(self.coraal_df), len(self.voc_df))
        else:
            return len(self.df)

if __name__ == '__main__':
    from args.cycleGAN_train_arg_parser import CycleGANTrainArgParser
    parser = CycleGANTrainArgParser()
    args = parser.parse_args()
    print(args)
    ds = Dataset(args, split='train', coraal=True, voc=True, return_pair=True)

    for item in ds:
        if len(item) == 2:
            item1, item2 = item
            print(item1)
            print(item2)
        else:
            print(item)
        break
