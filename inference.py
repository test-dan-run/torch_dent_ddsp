import torch
import librosa
import numpy as np

import warnings
warnings.simplefilter('ignore', UserWarning)

import hydra
from omegaconf import DictConfig
import soundfile as sf

from src.orchestrator import LightningDENT

EXAMPLE_WAV_PATH = 'ms_MY_144155_20200616_Left_000.wav'
OUTPUT_WAV_PATH = 'ms_MY_144155_20200616_Left_000_out2.wav'
CHECKPOINT_PATH = 'outputs/2023-02-07/10-38-05/logs/lightning_logs/version_0/checkpoints/best.ckpt'

@hydra.main(config_path='configs', config_name='test')
def main(cfg: DictConfig):

    # # setup dataset
    # data_module = LightningWavPairDataset(
    #     cfg.dataset, batch_size=cfg.run.batch_size,
    #     )
    # # data_module.prepare_data()

    #########
    # TRAIN #
    #########

    # initialise model
    model = LightningDENT.load_from_checkpoint(CHECKPOINT_PATH, cfg=cfg.model).to('cuda:0')
    model.eval()

    input_signal, _ = librosa.load(EXAMPLE_WAV_PATH, sr=16000)
    input_signal[np.where(input_signal==0)] = 1e-12
    input_tensor = torch.from_numpy(input_signal)

    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output_tensor = model(input_tensor.to('cuda:0'))
    output_tensor = output_tensor.squeeze()
    output_tensor = output_tensor.detach().cpu().numpy()

    sf.write(OUTPUT_WAV_PATH, output_tensor, 16000)

if __name__ == '__main__':
    main()