import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import  Any

from .utils import clip_by_value, frequency_filter

class FilteredNoise(torch.nn.Module):
    def __init__(
        self, window_size: int = 257, scale_fn: Any = clip_by_value, 
        bias_init: float = None, magnitude_init: float = None, 
        n_frequency_bins: int = None, n_frames: int = None):

        super(FilteredNoise, self).__init__()

        self.window_size = window_size
        self.scale_fn = scale_fn
        self.n_frequency_bins = n_frequency_bins
        self.n_frames = n_frames

        if magnitude_init is None:
            if self.n_frames is None:
                size = (1, self.n_frequency_bins)
            else:
                size = (1, self.n_frames, self.n_frequency_bins)
            magnitude_init = torch.normal(mean=0.0, std=0.05, size=size)
            
        self.magnitudes = Variable(magnitude_init, requires_grad=True)

        if bias_init is None:
            bias_init = torch.normal(mean=0.0, std=0.05, size=(1,))
        self.bias = Variable(bias_init, requires_grad=True)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:

        if self.scale_fn:
            magnitudes = self.scale_fn(self.magnitudes + self.bias)
        else:
            magnitudes = self.magnitudes + self.bias
        
        noise = nn.init.uniform_(torch.empty(size=audio.size()), a=-0.0, b=1.0)

        return frequency_filter(noise, magnitudes, window_size=self.window_size)
