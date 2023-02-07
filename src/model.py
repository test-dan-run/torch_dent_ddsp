import torch
from typing import Dict

from src.core import DynamicRangeCompressor, FIRFilter, FilteredNoise, Distortion

class DENT(torch.nn.Module):
    def __init__(
        self,
        waveshaper_config: Dict = None,
        compressor_config: Dict = None,
        equalizer_config: Dict = None,
        noise_config: Dict = None,
        noise_adjustment: float = 0.0,
        ):
        super(DENT, self).__init__()

        self.waveshaper = Distortion(**waveshaper_config) if waveshaper_config is not None else None
        self.compressor = DynamicRangeCompressor(**compressor_config) if compressor_config is not None else None
        self.equalizer = FIRFilter(**equalizer_config) if equalizer_config is not None else None
        self.noise = FilteredNoise(**noise_config) if noise_config is not None else None
        self.noise_adjustment = noise_adjustment 

    def __call__(self, audio: torch.Tensor) -> torch.Tensor: 
        output = audio
        if self.waveshaper:
            output = self.waveshaper(output)
        if self.compressor:
            output = self.compressor(output)
        if self.equalizer:
            output = self.equalizer(output)
        if self.noise:
            noise = self.noise(output)
            if self.noise_adjustment > 0.0:
                adj_linear = 20 ** (self.noise_adjustment/20)
                noise *= adj_linear
            output = output + noise

        return output

if __name__ == '__main__':

    dent = DENT(
        waveshaper_config = {
            'threshold': 15.0,
        },
        compressor_config = {
            'sample_rate': 16000,
            'threshold': -10,
            'ratio': 30.0,
            'makeup': 0.0,
            'attack': 1.0e-7,
            'release': 1.0e-3,
            'downsample_factor': 2.0
        },
        equalizer_config = {
            'n_frequency_bins': 1000,
        },
        noise_config = {
            'n_frequency_bins': 1000
        }
    )

    batch_tensor = torch.rand(size=(4, 1, 16000))
    with torch.no_grad():
        out = dent(batch_tensor)
    print(out.shape)
    print(out)