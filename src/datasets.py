import os
import random
import numpy as np

import json
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset

from typing import Union, Tuple, List

class WavDataset(Dataset):
    def __init__(
        self, manifest_path: str, labels_list: List[List[str]] = None, labels_indices_path: str = None,
        sample_rate: int = 44100, n_fft: int = 1024, hop_length: int = 512,
        power: float = 2.0, num_mels: int = 128,
        audio_length_sec: Union[int, float] = 10, random_seed: int = 42):
        super(AudioSetDataset, self).__init__()

        assert labels_list is not None or labels_indices_path is not None, 'Both labels_list and labels_indices_path cannot be None'
        if labels_list and labels_indices_path:
            print('Both labels_list and labels_indices_path are not None. Defaulting to using labels_list.')

        random.seed(random_seed)
        self.sample_rate = sample_rate
        self.sample_length = audio_length_sec * sample_rate

        with open(manifest_path, mode='r', encoding='utf-8') as fr:
            lines = fr.readlines()
        self.items = [json.loads(line.strip('\r\n')) for line in lines]

        if labels_indices_path and not labels_list:
            with open(labels_indices_path, mode='r', encoding='utf-8') as fr:
                lines = fr.readlines()
            # [[idx1, AudioSetlabel1, Textlabel1], [idx2, AudioSetlabel2, Textlabel2], ...]
            labels_list = [line.strip('\n').split(',', 2) for line in lines]
        
        # [AudioSetlabel1, AudioSetlabel2, ...]
        self.idx2aslabel = [line[1] for line in labels_list]
        # {AudioSetlabel1: idx1, AudioSetlabel2: idx2, ...}
        self.aslabel2idx = {line[1]: idx for idx, line in enumerate(labels_list)}
        # [Textlabel1, Textlabel2, ...]
        self.idx2textlabel = [line[2].replace('"', '') for line in labels_list]
        # assuming 5 classes -> [Tensor([1,0,0,0,0]), Tensor([0,1,0,0,0]), ..., Tensor([0,0,0,0,1])]
        self.idx2y = [torch.FloatTensor(np.eye(len(self.idx2aslabel), dtype='uint8')[i]) for i in range(len(self.idx2aslabel))]

        # transformation initialisation
        self.melspec_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
            n_mels=num_mels
            )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx.tolist()

        ### PROCESS INPUTS ###         
        input_signal, _ = torchaudio.load(self.items[idx]['audio_filepath'], normalize=True)

        # pad audio if insufficient length
        if input_signal.size(1) < self.sample_length:
            input_signal = torch.nn.functional.pad(input_signal, (0, self.sample_length - input_signal.size(1)), 'constant')

        # truncate audio if more than req. length
        elif input_signal.size(1) > self.sample_length:
            input_signal = input_signal[:, :self.sample_length]

        melspec = self.melspec_transform(input_signal)
        # log_offset -> 1e-7
        log_melspec = torch.log(melspec + 1e-7)

        ### PROCESS OUTPUTS ###
        aslabels = self.items[idx]['labels']
        label_indices = [self.aslabel2idx[l] for l in aslabels]
        target = sum([self.idx2y[idx] for idx in label_indices])

        return log_melspec, target

    def __len__(self) -> int:
        return len(self.items)