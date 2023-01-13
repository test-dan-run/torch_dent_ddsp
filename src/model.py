import torch
from typing import Dict

from .core import DynamicRangeCompressor, FIRFilter, FilteredNoise, Distortion

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
