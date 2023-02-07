import os
import json
import random
from typing import Tuple, Optional

import torch
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

import librosa
import numpy as np
from scipy import signal
from pydub import AudioSegment
from omegaconf import DictConfig

class WavPairDataset(Dataset):
    def __init__(self, manifest_path: str, random_seed: int = 42):
        super(WavPairDataset, self).__init__()

        random.seed(random_seed)

        with open(manifest_path, mode='r', encoding='utf-8') as fr:
            lines = fr.readlines()
        self.items = [json.loads(line.strip('\r\n')) for line in lines]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx.tolist()

        ### PROCESS INPUTS ###
        input_signal = torch.load(self.items[idx]['audio_filepath'])
        output_signal = torch.load(self.items[idx]['output_audio_filepath'])

        return input_signal, output_signal

    def __len__(self) -> int:
        return len(self.items)
    
class LightningWavPairDataset(pl.LightningModule):
    def __init__(self, cfg: DictConfig, batch_size: int = 32):
        super(LightningWavPairDataset, self).__init__()
        self.cfg = cfg
        self.batch_size = batch_size

    def prepare_data(self):

        def prepare_set(manifest_path: str, new_dir: str = 'prepped_data') -> str:

            with open(manifest_path, mode='r', encoding='utf-8') as fr:
                lines = fr.readlines()
            items = [json.loads(line.strip('\r\n')) for line in lines]

            base_dir = os.path.dirname(manifest_path)
            output_dir = os.path.join(base_dir, new_dir)
            input_tensor_dir = os.path.join(output_dir, 'clean', 'tensor')
            output_tensor_dir = os.path.join(output_dir, 'noisy', 'tensor')
            os.makedirs(input_tensor_dir, exist_ok=True)
            os.makedirs(output_tensor_dir, exist_ok=True)

            new_manifest_path = os.path.join(output_dir, 'updated_' + os.path.basename(manifest_path))
            with open(new_manifest_path, mode='w', encoding='utf-8') as fw:
                for item in items:

                    ### PROCESS INPUTS ###
                    input_signal, _ = librosa.load(os.path.join(base_dir, item['audio_filepath']), sr=16000)
                    output_signal, _ = librosa.load(os.path.join(base_dir, item['output_audio_filepath']), sr=16000)

                    # normalize noisy signal
                    input_dbfs = AudioSegment.from_wav(os.path.join(base_dir, item['audio_filepath'])).dBFS
                    output_dbfs = AudioSegment.from_wav(os.path.join(base_dir, item['output_audio_filepath'])).dBFS
                    change_in_db = input_dbfs - output_dbfs
                    change_in_float = 10 ** (change_in_db / 20)
                    output_signal = output_signal * change_in_float

                    # align signals
                    if len(input_signal) != len(output_signal):
                        zeros_array = np.zeros(( abs(len(input_signal)-len(output_signal)), ), dtype=np.float32)
                        if len(input_signal) < len(output_signal):
                            input_signal = np.append(input_signal, zeros_array)
                        else:
                            output_signal = np.append(output_signal, zeros_array)
                    
                    corr = signal.correlate(input_signal, output_signal, mode='full')
                    lags_arr = signal.correlation_lags(len(input_signal), len(output_signal), mode='full')
                    lag = lags_arr[np.argmax(corr)]

                    # shift the array, return aligned and equal length array
                    if lag>0:
                        output_padded = np.pad(output_signal, (lag, 0), 'constant', constant_values=0)#shift right, pad zero left
                        output_signal_shifted = output_padded[:len(output_signal)] 
                    elif lag<0:
                        output_padded = np.pad(output_signal, (0, abs(lag)), 'constant', constant_values=0)#shift left, pad zero right
                        output_signal_shifted = output_padded[abs(lag):]
                    else:
                        output_signal_shifted = output_signal
                
                    if len(input_signal) % 2 == 1:
                        input_signal = np.pad(input_signal, (0, 1), 'constant', constant_values=0)
                        output_signal_shifted = np.pad(output_signal_shifted, (0, 1), 'constant', constant_values=0)
                    
                    # noise floor
                    input_signal[np.where(input_signal==0)]=1e-12
                    output_signal_shifted[np.where(output_signal_shifted==0)]=1e-12

                    torch_input = torch.from_numpy(input_signal)
                    torch_output = torch.from_numpy(output_signal_shifted)

                    assert torch_input.shape == torch_output.shape

                    if torch_input.dim() == 1:
                        torch_input = torch_input.unsqueeze(0)
                    if torch_output.dim() == 1:
                        torch_output = torch_output.unsqueeze(0)

                    out_input_filename = os.path.splitext(os.path.basename(item['audio_filepath']))[0]
                    out_input_path = os.path.join(input_tensor_dir, out_input_filename + '.pt')
                    out_output_filename = os.path.splitext(os.path.basename(item['output_audio_filepath']))[0]
                    out_output_path = os.path.join(output_tensor_dir, out_output_filename + '.pt')

                    torch.save(torch_input, out_input_path)
                    torch.save(torch_output, out_output_path)

                    new_item = {
                        'audio_filepath': out_input_path,
                        'output_audio_filepath': out_output_path
                    }

                    fw.write(json.dumps(new_item)+'\n')

            return new_manifest_path
        
        self.cfg.train_manifest_path = prepare_set(self.cfg.train_manifest_path)
        self.cfg.valid_manifest_path = prepare_set(self.cfg.valid_manifest_path)
        if self.cfg.test_manifest_path:
            self.cfg.test_manifest_path = prepare_set(self.cfg.test_manifest_path)
    
    @staticmethod
    def collate_override(batch):

        input_list, output_list = [],[]
        max_size = max(batch, key=lambda x: x[0].size(-1))[0].size(-1)
        for (_inp, _out) in batch:
            if _inp.size(-1) < max_size:
                _inp = F.pad(_inp, (0, max_size-_inp.size(-1)))
            if _out.size(-1) < max_size:
                _out = F.pad(_out, (0, max_size-_out.size(-1)))            
            input_list.append(_inp)
            output_list.append(_out)

        return torch.stack(input_list), torch.stack(output_list)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.train_data = WavPairDataset(
                manifest_path = self.cfg.train_manifest_path, 
                random_seed = self.cfg.random_seed,
            )

            self.valid_data = WavPairDataset(
                manifest_path = self.cfg.valid_manifest_path, 
                random_seed = self.cfg.random_seed,
            )

        if stage == 'test':
            self.test_data = WavPairDataset(
                manifest_path = self.cfg.test_manifest_path, 
                random_seed = self.cfg.random_seed,
            )


    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size, collate_fn=LightningWavPairDataset.collate_override)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, batch_size=self.batch_size, collate_fn=LightningWavPairDataset.collate_override)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=self.batch_size, collate_fn=LightningWavPairDataset.collate_override)