import os
import json
import torch
import shutil
import librosa
import numpy as np
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore', UserWarning)

import hydra
from omegaconf import DictConfig
import soundfile as sf

from src.orchestrator import LightningDENT

@hydra.main(config_path='configs', config_name='batch_inference')
def main(cfg: DictConfig):

    input_manifest_path = os.path.join(cfg.dataset.input_manifest_dir, cfg.dataset.manifest_name)
    assert os.path.exists(input_manifest_path), f'Manifest path [{input_manifest_path}] do not exist.'

    with open(input_manifest_path, mode='r') as f:
        lines = f.readlines()
    items = [json.loads(line) for line in lines]

    os.makedirs(cfg.dataset.output_manifest_dir, exist_ok=True)
    # initialise model
    model = LightningDENT.load_from_checkpoint(cfg.model.checkpoint_path, cfg=cfg.model).to('cuda:0')
    model.eval()

    for item in tqdm(items):

        input_path = os.path.join(cfg.dataset.input_manifest_dir, item['audio_filepath'])
        output_path = os.path.join(cfg.dataset.output_manifest_dir, item['audio_filepath'])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        input_signal, _ = librosa.load(input_path, sr=16000)
        input_signal[np.where(input_signal==0)] = 1e-12
        input_tensor = torch.from_numpy(input_signal)

        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output_tensor = model(input_tensor.to('cuda:0'))
        output_tensor = output_tensor.squeeze()
        output_tensor = output_tensor.detach().cpu().numpy()

        sf.write(output_path, output_tensor, 16000)

if __name__ == '__main__':
    main()